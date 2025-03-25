import ctranslate2
import librosa
import transformers
import numpy as np
import webrtcvad
import wave
import collections
import os
import subprocess
from test_vad import detect_voice_activity, preprocess_audio

# Load and resample the audio file.
audio_path = "test.wav"
# Sử dụng VAD để phát hiện và chia các đoạn giọng nói
voice_chunks = detect_voice_activity(
    audio_path,
    aggressiveness=3,
    frame_duration_ms=20,
    padding_duration_ms=500,
    max_chunk_ms=15000
)

# Khởi tạo processor
processor = transformers.WhisperProcessor.from_pretrained("erax-ai/EraX-WoW-Turbo-V1.0")

# Load the model on CPU.
model = ctranslate2.models.Whisper("whisper-tiny-ct2")

transcription_full = ""

# Xử lý từng chunk giọng nói
for i, chunk_info in enumerate(voice_chunks):
    # Chuyển chunk bytes thành numpy array
    chunk = np.frombuffer(chunk_info['chunk'], dtype=np.int16).astype(np.float32) / 32768.0
    
    # Compute features
    inputs = processor(chunk, return_tensors="np", sampling_rate=16000)
    features = ctranslate2.StorageView.from_array(inputs.input_features)
    
    # Detect language (chỉ cần làm cho đoạn đầu tiên)
    if i == 0:
        results = model.detect_language(features)
        language, probability = results[0][0]
        print("Detected language %s with probability %f" % (language, probability))
        
        prompt = processor.tokenizer.convert_tokens_to_ids([
            "<|startoftranscript|>",
            language,
            "<|transcribe|>",
            "<|notimestamps|>",
        ])
    
    # Run generation for the current chunk
    results = model.generate(features, [prompt])
    chunk_text = processor.decode(results[0].sequences_ids[0])
    
    # Loại bỏ các token đặc biệt từ kết quả nếu không phải đoạn đầu tiên
    if i > 0:
        for token in ["<|startoftranscript|>", language, "<|transcribe|>", "<|notimestamps|>"]:
            if token in chunk_text:
                chunk_text = chunk_text.replace(token, "")
    
    transcription_full += chunk_text + " "
    print(f"Processed chunk {i+1}/{len(voice_chunks)}")

print("\nFull transcription:")
print(transcription_full)


"""
Vậy thì hiện tại thì là quá trình của mình như thế nào em? Quá trình làm việc trong ngành công nghệ thông tin này như thế nào em?
Dạ để em xem, em đi làm công ty 2 3, em đi làm 5 công ty rồi, đang làm 5.5. 5 công ty rồi quả?
Dạ vâng, tụi trẻ. Thì đầu tiên em năm 2 thì em đi xin việc cũng vất vả lắm, lúc đấy em đi xin intern. Lúc đấy là em học NS thì internet và ri app. Về mấy cái kiến thức phụ thì em đã biết triển khai, triển khai cơ bản hỏi là mình chưa hiểu sâu về tách của nó. Nhưng mà triển khai giống như 'Dốc Cơ, Rabit Kuel, Reddit, Shukana là em đã biết triển khai cơ bản rồi nhưng mà em vẫn chưa hiểu sâu. Lúc đó em đi xin thực tập. Xin thực tập năm 2 thì em gửi CV, em gửi 3-4 ngày là em gửi lại tìm em gửi. Thì gửi trung bình tầm 3-4 ngày thì 10 công ty, 3-4 ngày gửi công ty. Em gửi một chạp, lúc em tỉnh thì hình như phải nộp gần.
"""