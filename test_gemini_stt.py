
from google import genai
import os
from dotenv import load_dotenv

from google.genai import types



load_dotenv()

with open('test1.wav', 'rb') as f:
    image_bytes = f.read()


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


prompt = """
Can you transcribe this interview, in the format of timecode, speaker, caption.
Use speaker A, speaker B, etc. to identify speakers.
"""


response = client.models.generate_content(
    model='gemini-2.0-flash-lite',
    contents=[
        prompt,
        types.Part.from_bytes(
        data=image_bytes,
        mime_type='audio/wav',
        )
    ]
)

print(response.text)
# Example response:
# [00:00:00] Speaker A: Your devices are getting better over time...
# [00:00:16] Speaker B: Welcome to the Made by Google podcast, ...
# [00:01:00] Speaker A: So many features. I am a singer. ...
# [00:01:33] Speaker B: Amazing. DeCarlos, same question to you, ...