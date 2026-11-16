# Training image: the TTS image (torch 2.9.1 cu128, sm_120) plus the VoxCPM repo for its LoRA script.
FROM raisa-voice-tts:latest
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
COPY VoxCPM /VoxCPM
# --no-deps keeps the image torch; the script only needs these.
RUN pip install --no-cache-dir --no-deps -e /VoxCPM && pip install --no-cache-dir argbind tensorboardX "datasets>=3,<4" einops inflect addict wetext soundfile librosa simplejson sortedcontainers torchcodec==0.8.1 modelscope
WORKDIR /VoxCPM
