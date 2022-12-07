import os
import io
import openai
import requests
import pyaudio
import wave
import stt
import time
import logging
import threading
import collections
import queue
import os.path
import numpy as np
import webrtcvad
from pynput.keyboard import Key, Listener
from scipy import signal
from halo import Halo
from dotenv import load_dotenv
from datetime import datetime

logging.basicConfig(level=20)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    while True:
        prompt = ""
        use_stt = False

        # def on_press(key):
        #     global use_stt
        #     try:
        #         if key.char == "y":
        #             use_stt = True
        #             return False
        #         if key.char == "n":
        #             return False
        #     except:
        #         pass

        value = input("Use speech to text? (y/n): ")

        if value not in ["y", "n"]:
            continue
        elif value == "y":
            use_stt = True

        if use_stt:
            prompt = speech_to_text()
        else:
            prompt = input("Write prompt: ")

        if prompt:
            response = prompt_chatgpt(prompt)
            print(response)
        text_to_speech(response)


def prompt_chatgpt(prompt):
    r = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=256)
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


def speech_to_text():
    model = stt.Model("stt_models/en.tflite")
    vad_audio = VADAudio(aggressiveness=3,
                         input_rate=16000)
    print("Listening (Ctrl+c to stop)...")
    frames = vad_audio.vad_collector()

    result = ""

    try:
        spinner = Halo(spinner='line')
        stream_context = model.createStream()
        wav_data = bytearray()
        for frame in frames:
            if frame is not None:
                spinner.start()
                logging.debug("streaming frame")
                stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            else:
                if spinner:
                    spinner.stop()
                logging.debug("end utterence")
                text = stream_context.finishStream()
                print("Recognized: %s" % text)
                result += text + "\n"
                stream_context = model.createStream()
        # listener.join()
    except KeyboardInterrupt:
        pass

    return result.strip()


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None:
            def callback(in_data): return self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS /
                              float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(
            self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        stt

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


if __name__ == "__main__":
    main()
