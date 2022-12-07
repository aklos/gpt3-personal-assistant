# ChatGPT Personal Assistant

Interact with ChatGPT vocally, using both STT (speech-to-text) and TTS (text-to-speech), with keyboard prompts as a fallback.

## Setup

- Create a virtual environment: `python3 -m venv venv`
- Activate the venv: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Create ".env" file in root folder and set `OPENAI_API_KEY`

## Usage

### 1. Start the TTS server (in separate terminal session)
```
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --list_models # To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits # To start a server
```

### 2. Run the script
```
python main.py
```