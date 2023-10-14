# import torch
# from transformers import pipeline
# from datasets import load_dataset, Audio

# speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")


# dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
# print(dataset)
# dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
# result = speech_recognizer(dataset[:4]["audio"])
# print(result)

from transformers import pipeline
generator = pipeline(task="automatic-speech-recognition", model="openai/whisper-medium")
print(generator("C:\\Users\\Rafael\\Documents\\GitHub\\stimulus-control-sdl2\\data\\0-Rafael\\responses\\nibo-P00-S01-B00-T00-C01-R01.wav"))