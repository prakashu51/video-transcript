@echo off
:: Usage: extract_audio.bat input_video.mp4
:: Output: input_video_name.wav (mono, 16kHz - optimized for Whisper)
if "%~1"=="" (
    echo Usage: extract_audio.bat input_video.mp4
    exit /b 1
)
set "OUTPUT_FILE=%~dpn1.wav"
ffmpeg -i "%~1" -ar 16000 -ac 1 -c:a pcm_s16le "%OUTPUT_FILE%"
echo Done! %OUTPUT_FILE% is ready for transcription.
