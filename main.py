import os
import io
import openai
import requests
import pyaudio
import wave
import whisper
import speech_recognition as sr
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    audio_model = whisper.load_model("base.en")

    print("Model loaded.\n")

    while True:
        prompt = speech_to_text(audio_model)
        if len(prompt):
            response = prompt_chatgpt(prompt)
        text_to_speech(response)


def prompt_chatgpt(prompt):
    r = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=96)
    return r["choices"][0]["text"].strip()


def text_to_speech(text):
    r = requests.get(
        "http://localhost:5002/api/tts?text=%s&speaker_id=p260&style_wav=" % (text))
    with wave.open(io.BytesIO(r.content), 'rb') as f:
        width = f.getsampwidth()
        channels = f.getnchannels()
        rate = f.getframerate()
    pa = pyaudio.PyAudio()
    pa_stream = pa.open(
        format=pyaudio.get_format_from_width(width),
        channels=channels,
        rate=rate,
        output=True
    )
    pa_stream.write(r.content)


def speech_to_text(audio_model):
    transcription = ['']
    record_timeout = 2
    phrase_timeout = 2
    temp_file = NamedTemporaryFile().name

    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()

    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout)

    print('Enter prompt...')

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                    if len(transcription) > 0:
                        break
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(
                    last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file)
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)
    return transcription[0]


if __name__ == "__main__":
    main()
