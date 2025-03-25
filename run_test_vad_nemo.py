import subprocess

# Định nghĩa lệnh cần chạy
command = [
    "python", "test_vad_nemo.py",
    "--config-path", "../conf/vad",
    "--config-name", "frame_vad_infer_postprocess",
    "input_manifest=manifest.json",
    "output_dir=output/vad_results"
]

# Chạy lệnh
result = subprocess.run(command, capture_output=True, text=True)

# In kết quả
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
