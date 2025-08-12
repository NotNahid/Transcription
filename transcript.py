# Install Whisper and dependencies
!pip install -q git+https://github.com/openai/whisper.git
!sudo apt-get install -y ffmpeg

import whisper
import os
from google.colab import files

# Upload audio file
print("📤 Upload your audio file (mp3, wav, m4a, etc):")
uploaded = files.upload()

audio_path = list(uploaded.keys())[0]
print(f"🎧 Audio file uploaded: {audio_path}")

# Load Whisper model (base is fast, tiny is faster but less accurate, large is slow but very accurate)
model = whisper.load_model("base")

# Transcribe with timestamps
print("⏳ Transcribing audio, please wait...")
result = model.transcribe(audio_path, verbose=True)

segments = result["segments"]  # list of segments with timestamps and text

# Convert seconds to SRT time format
def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Write SRT file
srt_filename = os.path.splitext(audio_path)[0] + ".srt"
with open(srt_filename, "w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

print(f"✅ Transcription complete! Subtitle file saved as: {srt_filename}")

# Download the SRT file
files.download(srt_filename)
