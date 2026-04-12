@echo off
:: Usage: extract_audio.bat input_video.mp4
:: Output: audio.wav (mono, 16kHz - optimized for Whisper)
if "%~1"=="" (
    echo Usage: extract_audio.bat input_video.mp4
    exit /b 1
)
ffmpeg -i "%~1" -ar 16000 -ac 1 -c:a pcm_s16le audio.wav
echo Done! audio.wav is ready for transcription.
