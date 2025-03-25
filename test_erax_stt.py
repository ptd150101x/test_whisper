import ctranslate2
import librosa
import numpy as np
from transformers import AutoProcessor

# Tải processor từ Hugging Face
processor = AutoProcessor.from_pretrained("erax-ai/EraX-WoW-Turbo-V1.0")

# Tải mô hình CTranslate2 đã chuyển đổi
model = ctranslate2.models.Whisper("whisper-tiny-ct2/", device="cpu", compute_type="float32")
# Sử dụng device="cuda" nếu bạn muốn chạy trên GPU
# compute_type có thể là "float32", "float16", "int8" tùy theo phần cứng của bạn

# Hàm tiền xử lý âm thanh
def load_audio(file_path, sample_rate=16000):
    # Tải và resample âm thanh đến 16kHz
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

# Hàm nhận dạng âm thanh
def transcribe_audio(audio_file, language=None, task="transcribe"):
    # Tải và xử lý âm thanh
    audio = load_audio(audio_file)
    
    # Lấy đặc trưng âm thanh
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
    features = inputs.input_features.astype(np.float32)
    
    # Chuẩn bị các tham số
    generation_config = {
        "beam_size": 5,
        "max_length": 448,
        "sampling_topk": 0,
        "sampling_temperature": 0,
    }
    
    if language:
        generation_config["language"] = language
    
    # Thực hiện suy luận
    output = model.generate(
        features,
        generation_config,
        task=task
    )
    
    # Giải mã kết quả
    results = processor.tokenizer.batch_decode(
        output.sequences_ids[0],
        skip_special_tokens=True,
        normalize=True
    )
    
    return results[0] if results else ""

# Sử dụng hàm
audio_file = "test.wav"
# Nhận dạng tiếng Việt
transcript = transcribe_audio(audio_file, language="vi", task="transcribe")
print(f"Kết quả nhận dạng: {transcript}")

# # Hoặc dịch sang tiếng Anh
# translation = transcribe_audio(audio_file, task="translate")
# print(f"Bản dịch sang tiếng Anh: {translation}")