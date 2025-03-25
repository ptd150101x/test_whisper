import os
import time
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.soniox.com"
 
# Retrieve the API key from environment variable (ensure SONIOX_API_KEY is set)
API_KEY = os.environ["SONIOX_API_KEY"]
 
# Create a requests session and set the Authorization header
session = requests.Session()
session.headers["Authorization"] = f"Bearer {API_KEY}"

# Hàm để log response API
def log_response(step, response):
    print(f"\n--- {step} Response ---")
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {json.dumps(dict(response.headers), indent=2, ensure_ascii=False)}")
    try:
        print(f"Response Body: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except:
        print(f"Response Body: {response.text[:500]}...")  # Chỉ hiển thị 500 ký tự đầu tiên nếu không phải JSON

# Bắt đầu đo thời gian tổng thể
start_time_total = time.time()


# 2. Start a new transcription session by sending the audio file to the API
print("\nStarting transcription...")
start_time = time.time()
response = session.post(
    f"{API_BASE}/v1/transcriptions",
    json={
        # "file_id": file["id"],
        "audio_url": "https://filebin.net/audui1rxnrdmlmdb/be-dev-m%e1%bb%9bi-ra-tr%c6%b0%e1%bb%9dng-03-n%c4%83m-kinh-nghi%e1%bb%87m---c%c3%b3-g%c3%ac-%c4%91%e1%ba%b7c-bi%e1%bb%87t.wav",
        # "audio_url": "https://filebin.net/audui1rxnrdmlmdb/be-dev-m%e1%bb%9bi-ra-tr%c6%b0%e1%bb%9dng-03-n%c4%83m-kinh-nghi%e1%bb%87m---c%c3%b3-g%c3%ac-%c4%91%e1%ba%b7c-bi%e1%bb%87t%20%281%29.wav",
        "model": "stt-async-preview",
        "enable_speaker_tags": True,
        "context": "Đây là buổi phỏng vấn trong ngành IT. Người phỏng vấn và ứng viên thảo luận về backend developer frontend developer fullstack developer JavaScript TypeScript React Angular Vue Node.js Express MongoDB SQL database REST API GraphQL Docker Kubernetes AWS Azure Google Cloud DevOps CI/CD Git GitHub GitLab microservices architecture agile scrum kanban unit testing integration testing performance testing debugging coding algorithms data structures design patterns object-oriented programming functional programming"
    },
)
transcription_start_time = time.time() - start_time
log_response("Transcription Start", response)
print(f"Transcription request completed in {transcription_start_time:.2f} seconds")

response.raise_for_status()
transcription = response.json()
 
transcription_id = transcription["id"]
print(f"Transcription started with ID: {transcription_id}")
 
# 3. Poll the transcription endpoint until the status is 'completed'
print("\nPolling for transcription completion...")
poll_start_time = time.time()
poll_count = 0

while True:
    poll_count += 1
    poll_iteration_start = time.time()
    
    response = session.get(f"{API_BASE}/v1/transcriptions/{transcription_id}")
    response.raise_for_status()
    transcription = response.json()
    
    poll_iteration_time = time.time() - poll_iteration_start
    print(f"\nPoll #{poll_count} completed in {poll_iteration_time:.2f} seconds")
    log_response(f"Poll #{poll_count}", response)
 
    status = transcription.get("status")
    if status == "error":
        raise Exception(
            f"Transcription error: {transcription.get('error_message', 'Unknown error')}"
        )
    elif status == "completed":
        # Stop polling when the transcription is complete
        break
 
    # Wait for 1 second before polling again
    time.sleep(1)

polling_time = time.time() - poll_start_time
print(f"\nPolling completed in {polling_time:.2f} seconds after {poll_count} attempts")
 
# 4. Retrieve the final transcript once transcription is completed
print("\nRetrieving final transcript...")
start_time = time.time()
response = session.get(f"{API_BASE}/v1/transcriptions/{transcription_id}/transcript")
transcript_time = time.time() - start_time
log_response("Final Transcript", response)
print(f"Transcript retrieval completed in {transcript_time:.2f} seconds")

response.raise_for_status()
transcript = response.json()
 
# Tính tổng thời gian
total_time = time.time() - start_time_total
print(f"\n--- Tổng kết thời gian ---")
print(f"Thời gian bắt đầu transcription: {transcription_start_time:.2f} giây")
print(f"Thời gian polling: {polling_time:.2f} giây ({poll_count} lần)")
print(f"Thời gian lấy transcript: {transcript_time:.2f} giây")
print(f"Tổng thời gian: {total_time:.2f} giây")

# Print the transcript text
print("\nTranscript:")
print(transcript["text"])