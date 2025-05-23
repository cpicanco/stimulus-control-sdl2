from openai import OpenAI
client = OpenAI()

audio_file= open("C:\\Users\\Rafael\\Documents\\GitHub\\stimulus-control-sdl2\\media\\wav\\rafael\\nibo.wav", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
print(transcript)