from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from pydub import AudioSegment
from io import BytesIO

# Get the key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Assign voices to roles
voices = {
    "agent": "alloy",
    "customer": "verse"
}

# Load dialogues from JSON file
with open("dialogues.json", "r") as f:
    dialogues = json.load(f)

# Loop through each dialogue
for i, dialogue in enumerate(dialogues, start=1):
    conversation = AudioSegment.silent(duration=500)

    for speaker, line in dialogue:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voices[speaker],
            input=line
        )
        audio = AudioSegment.from_file(BytesIO(response.read()), format="mp3")
        conversation += audio + AudioSegment.silent(duration=300)

    # Export each dialogue as its own WAV file
    filename = f"call_center_dialogue_{i}.wav"
    conversation.export(filename, format="wav")
    print(f"✅ Saved {filename}")