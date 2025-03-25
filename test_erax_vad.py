import os
import wave
import asyncio
import ctranslate2
import librosa
import numpy as np
import webrtcvad
import collections
import subprocess
from transformers import AutoProcessor
from io import BytesIO

# Tải processor từ Hugging Face
processor = AutoProcessor.from_pretrained("erax-ai/EraX-WoW-Turbo-V1.0")

# Tải mô hình CTranslate2 đã chuyển đổi
model = ctranslate2.models.Whisper("whisper-tiny-ct2/", device="cpu", compute_type="int8")

# Hàm tiền xử lý âm thanh
def load_audio(audio_data, sample_rate=16000):
    # Nếu là đường dẫn file
    if isinstance(audio_data, str):
        audio, sr = librosa.load(audio_data, sr=sample_rate)
    # Nếu là bytes data
    elif isinstance(audio_data, bytes):
        with BytesIO(audio_data) as buffer:
            audio, sr = librosa.load(buffer, sr=sample_rate)
    return audio

async def transcribe_audio_chunk(audio_data, language=None, task="transcribe"):
    """Nhận dạng một chunk âm thanh"""
    # Tải và xử lý âm thanh
    audio = load_audio(audio_data)
    
    # Lấy đặc trưng âm thanh
    inputs = processor(audio, return_tensors="np", sampling_rate=16000)
    features = ctranslate2.StorageView.from_array(inputs.input_features)
    
    # Phát hiện ngôn ngữ nếu không được chỉ định
    if language is None:
        results = model.detect_language(features)
        language, probability = results[0][0]
        print(f"Detected language {language} with probability {probability}")
    
    # Tạo prompt
    prompt = processor.tokenizer.convert_tokens_to_ids([
        "<|startoftranscript|>",
        language,
        "<|transcribe|>" if task == "transcribe" else "<|translate|>",
        "<|notimestamps|>",
    ])
    
    # Thực hiện suy luận (chạy trong executor để không chặn event loop)
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: model.generate(features, [prompt])
    )
    
    # Giải mã kết quả
    transcript = processor.decode(results[0].sequences_ids[0])
    
    # Loại bỏ các token đặc biệt
    for token in ["<|startoftranscript|>", language, "<|transcribe|>", "<|translate|>", "<|notimestamps|>"]:
        if token in transcript:
            transcript = transcript.replace(token, "")
    
    return transcript.strip()

def preprocess_audio_in_memory(input_path):
    """
    Preprocessing audio file trong bộ nhớ:
    - Chuyển sang mono
    - Chuyển sang 16-bit PCM
    - Sample rate 16kHz
    """
    try:
        # Đọc file âm thanh bằng librosa
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        
        # Chuyển thành bytes
        with BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                
                # Chuyển numpy array thành bytes
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                wf.writeframes(audio_bytes)
            
            # Lấy bytes từ buffer
            wav_buffer.seek(0)
            processed_audio = wav_buffer.read()
            
        return processed_audio, 16000
    
    except Exception as e:
        print(f"Lỗi preprocessing: {e}")
        return None, None

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Chia audio thành các frame"""
    n_bytes_per_sample = 2  # 16-bit audio
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * n_bytes_per_sample)
    
    for start_index in range(0, len(audio), bytes_per_frame):
        yield audio[start_index:start_index + bytes_per_frame]

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames, max_chunk_ms=15000):
    """Thu thập các khung có hoạt động giọng nói"""
    # Tính số frame padding dựa trên padding_duration_ms
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # Sử dụng deque với kích thước cố định để lưu trữ các frame gần nhất
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # Biến đánh dấu trạng thái đang thu thập giọng nói
    triggered = False
    # Danh sách lưu trữ các frame có giọng nói
    voiced_frames = []
    # Danh sách lưu trữ các chunk giọng nói đã phát hiện
    voice_chunks = []
    # Vị trí frame bắt đầu của chunk hiện tại
    current_chunk_start = 0
    # Thời lượng hiện tại của chunk
    current_chunk_duration = 0
    
    # Lưu trữ các điểm lặng tiềm năng để cắt chunk dài
    silence_points = []
    
    # Ngưỡng phát hiện giọng nói (có thể điều chỉnh)
    speech_threshold = 0.8  # 80% frame trong buffer là giọng nói
    silence_threshold = 0.9  # 90% frame trong buffer là im lặng
    
    # Duyệt qua từng frame
    for frame_index, frame in enumerate(frames):
        # Kiểm tra kích thước frame hợp lệ (cho 20ms ở 16kHz)
        expected_frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
        if len(frame) < expected_frame_size:
            continue
            
        try:
            # Kiểm tra frame có chứa giọng nói không
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception as e:
            print(f"Lỗi xử lý frame {frame_index}: {e}")
            continue
        
        # Thêm frame vào buffer
        ring_buffer.append((frame, is_speech))
        
        if not triggered:
            # Đang ở trạng thái im lặng, kiểm tra xem có nên bắt đầu thu thập không
            num_voiced = len([f for f, speech in ring_buffer if speech])
            speech_ratio = num_voiced / len(ring_buffer)
            
            # Nếu tỷ lệ giọng nói vượt ngưỡng, bắt đầu thu thập
            if speech_ratio > speech_threshold:
                triggered = True
                current_chunk_start = max(0, frame_index - len(ring_buffer) + 1)
                # Thêm các frame trong buffer vào danh sách voiced_frames
                voiced_frames.extend(f for f, speech in ring_buffer)
                current_chunk_duration = len(voiced_frames) * frame_duration_ms
                ring_buffer.clear()
                silence_points = []
        else:
            # Đang ở trạng thái thu thập, thêm frame hiện tại
            voiced_frames.append(frame)
            current_chunk_duration += frame_duration_ms
            
            # Kiểm tra xem có nên kết thúc thu thập không
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            silence_ratio = num_unvoiced / len(ring_buffer)
            
            # Ghi nhận điểm có khoảng lặng nhẹ (nhưng chưa đủ để kết thúc chunk)
            if num_unvoiced > 0.5 * len(ring_buffer) and num_unvoiced < silence_threshold * len(ring_buffer):
                silence_points.append({
                    'frame_index': frame_index,
                    'duration_so_far': current_chunk_duration,
                    'voiced_frames_index': len(voiced_frames)
                })
                
                # Chỉ giữ lại các điểm lặng gần đây nhất
                if len(silence_points) > 5:
                    silence_points = silence_points[-5:]
            
            # Kiểm tra điều kiện kết thúc chunk
            if silence_ratio > silence_threshold:
                # Kết thúc một chunk giọng nói do phát hiện khoảng lặng
                chunk = b''.join(voiced_frames)
                
                voice_chunks.append({
                    'chunk': chunk,
                    'start_frame': current_chunk_start,
                    'end_frame': frame_index,
                    'duration_ms': current_chunk_duration
                })
                
                # Reset các biến
                voiced_frames = []
                triggered = False
                ring_buffer.clear()
                current_chunk_duration = 0
                silence_points = []
            elif current_chunk_duration >= max_chunk_ms:
                # Vượt quá độ dài tối đa, tìm điểm lặng gần nhất để cắt
                if silence_points:
                    # Lấy điểm lặng gần nhất
                    cut_point = silence_points[-1]
                    cut_index = cut_point['voiced_frames_index']
                    
                    # Tạo chunk đến điểm cắt
                    chunk = b''.join(voiced_frames[:cut_index])
                    voice_chunks.append({
                        'chunk': chunk,
                        'start_frame': current_chunk_start,
                        'end_frame': cut_point['frame_index'],
                        'duration_ms': cut_point['duration_so_far']
                    })
                    
                    # Giữ lại phần còn lại cho chunk tiếp theo
                    remaining_frames = voiced_frames[cut_index:]
                    voiced_frames = remaining_frames
                    current_chunk_start = cut_point['frame_index']
                    current_chunk_duration = len(remaining_frames) * frame_duration_ms
                    silence_points = []
                else:
                    # Không tìm thấy điểm lặng, buộc phải cắt tại vị trí hiện tại
                    chunk = b''.join(voiced_frames)
                    voice_chunks.append({
                        'chunk': chunk,
                        'start_frame': current_chunk_start,
                        'end_frame': frame_index,
                        'duration_ms': current_chunk_duration
                    })
                    
                    # Reset
                    voiced_frames = []
                    triggered = False
                    ring_buffer.clear()
                    current_chunk_duration = 0
                    silence_points = []

    # Xử lý chunk cuối cùng nếu còn
    if voiced_frames:
        chunk = b''.join(voiced_frames)
        
        voice_chunks.append({
            'chunk': chunk,
            'start_frame': current_chunk_start,
            'end_frame': len(frames) - 1,
            'duration_ms': current_chunk_duration
        })

    return voice_chunks

def merge_short_chunks(chunks, frame_duration_ms, min_chunk_duration=300, max_silence_duration=500, max_chunk_ms=15000):
    """Hợp nhất các chunk quá ngắn hoặc quá gần nhau, đảm bảo không vượt quá max_chunk_ms"""
    if not chunks:
        return []
    
    result = [chunks[0]]
    
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]
        last_chunk = result[-1]
        
        # Tính khoảng cách giữa chunk hiện tại và chunk trước đó
        silence_duration = (current_chunk['start_frame'] - last_chunk['end_frame']) * frame_duration_ms
        
        # Tính độ dài sau khi merge
        merged_duration = last_chunk['duration_ms'] + current_chunk['duration_ms']
        
        # Chỉ merge nếu thỏa mãn các điều kiện:
        # 1. Chunk hiện tại quá ngắn hoặc khoảng cách quá nhỏ
        # 2. Độ dài sau khi merge không vượt quá max_chunk_ms
        if (current_chunk['duration_ms'] < min_chunk_duration or silence_duration < max_silence_duration) and merged_duration <= max_chunk_ms:
            # Hợp nhất chunk
            merged_chunk = {
                'chunk': last_chunk['chunk'] + current_chunk['chunk'],
                'start_frame': last_chunk['start_frame'],
                'end_frame': current_chunk['end_frame'],
                'duration_ms': merged_duration
            }
            result[-1] = merged_chunk
        else:
            # Thêm chunk mới vào kết quả
            result.append(current_chunk)
    
    return result

def detect_voice_activity(audio_path, aggressiveness=3, frame_duration_ms=20, padding_duration_ms=500, max_chunk_ms=15000):
    """Phát hiện hoạt động giọng nói"""
    # Preprocessing audio trong bộ nhớ
    audio_data, sample_rate = preprocess_audio_in_memory(audio_path)
    if not audio_data:
        print("Lỗi xử lý file âm thanh")
        return []
    
    # Khởi tạo VAD với độ nhạy phù hợp (0-3)
    vad = webrtcvad.Vad(aggressiveness)
    
    # Tạo generator frame với kích thước phù hợp (10, 20 hoặc 30ms)
    frames = list(frame_generator(frame_duration_ms, audio_data, sample_rate))
    
    # Thu thập các chunk giọng nói
    voice_chunks = vad_collector(
        sample_rate, 
        frame_duration_ms, 
        padding_duration_ms, 
        vad, 
        frames,
        max_chunk_ms
    )
    
    # Hợp nhất các chunk ngắn, đảm bảo không vượt quá max_chunk_ms
    merged_chunks = merge_short_chunks(voice_chunks, frame_duration_ms, 
                                      min_chunk_duration=300, 
                                      max_silence_duration=500, 
                                      max_chunk_ms=max_chunk_ms)
    
    return merged_chunks

async def transcribe_chunk(chunk_data, language="vi", task="transcribe"):
    """Nhận dạng một chunk âm thanh"""
    # Tạo file WAV tạm thời trong bộ nhớ
    with BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(chunk_data)
        
        # Đặt con trỏ về đầu buffer
        wav_buffer.seek(0)
        
        # Thực hiện nhận dạng
        transcript = await transcribe_audio_chunk(wav_buffer.read(), language, task)
        
    return transcript

async def transcribe_audio_with_vad(audio_path, language="vi", task="transcribe"):
    """Nhận dạng âm thanh với VAD và xử lý bất đồng bộ"""
    # Phát hiện các chunk giọng nói
    voice_chunks = detect_voice_activity(audio_path)
    
    if not voice_chunks:
        print("Không phát hiện giọng nói trong file âm thanh")
        return ""
    
    # Tạo danh sách các task nhận dạng
    transcription_tasks = []
    for chunk_info in voice_chunks:
        task_obj = asyncio.create_task(
            transcribe_chunk(chunk_info['chunk'], language, task)
        )
        transcription_tasks.append((chunk_info, task_obj))
    
    # Chờ tất cả các task hoàn thành
    results = []
    for chunk_info, task_obj in transcription_tasks:
        transcript = await task_obj
        results.append({
            'start_ms': chunk_info['start_frame'] * 20,  # Giả sử frame_duration_ms=20
            'end_ms': chunk_info['end_frame'] * 20,
            'duration_ms': chunk_info['duration_ms'],
            'transcript': transcript
        })
    
    # Sắp xếp kết quả theo thời gian bắt đầu
    results.sort(key=lambda x: x['start_ms'])
    
    # Ghép kết quả
    full_transcript = " ".join(result['transcript'] for result in results)
    
    return full_transcript, results

# Hàm chính để sử dụng
async def main():
    audio_path = "test.wav"
    
    print("Đang xử lý file âm thanh...")
    full_transcript, chunk_results = await transcribe_audio_with_vad(audio_path)
    
    print("\nKết quả nhận dạng đầy đủ:")
    print(full_transcript)
    
    print("\nKết quả chi tiết theo chunk:")
    for i, result in enumerate(chunk_results, 1):
        print(f"Chunk {i}:")
        print(f"  Thời gian: {result['start_ms']}ms - {result['end_ms']}ms")
        print(f"  Thời lượng: {result['duration_ms']}ms")
        print(f"  Nội dung: {result['transcript']}")

if __name__ == "__main__":
    asyncio.run(main())