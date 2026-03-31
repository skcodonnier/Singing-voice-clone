# Singing Voice Clone

Clone any song into your own voice using [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) — a zero-shot singing voice conversion (SVC) pipeline.

Provide a short recording of your voice and a target song, and the pipeline will:
1. Preprocess your voice (extract F0 pitch contour)
2. Separate vocals from the song's instrumental
3. Run voice conversion using a 698M-parameter diffusion model
4. Mix your converted vocals back with the original instrumental

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) (for audio mixing)
- CUDA GPU recommended (CPU works but is much slower)

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/skcodonnier/Singing-voice-clone.git
cd Singing-voice-clone
```

### 2. Clone SoulX-Singer

```bash
git clone https://github.com/Soul-AILab/SoulX-Singer.git
```

### 3. Install dependencies

```bash
conda create -n soulxsinger python=3.10 -y
conda activate soulxsinger
pip install -r SoulX-Singer/requirements.txt
```

### 4. Download pretrained models

```bash
pip install -U huggingface_hub

# SVC model
huggingface-cli download Soul-AILab/SoulX-Singer --local-dir SoulX-Singer/pretrained_models/SoulX-Singer

# Preprocessing models
huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess
```

## Usage

```bash
python convert.py --voice DATA/your_voice.wav --song DATA/song.wav
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--voice` | *(required)* | Path to your voice recording (WAV, max 30s used) |
| `--song` | *(required)* | Path to the target song (WAV/MP3) |
| `--steps` | `4` | Diffusion steps: 4=fast, 8=balanced, 32=best quality |
| `--cfg` | `1.0` | CFG scale (1.0–3.0, higher = more timbre similarity) |
| `--pitch` | `0` | Pitch shift in semitones |
| `--device` | auto | `cuda:0` or `cpu` (auto-detected) |

### Example

```bash
python convert.py --voice DATA/my_voice.wav --song "Ed_Sheeran_Perfect.wav" --steps 8
```

## Output

Results are saved to `audio_output/<songname>_<timestamp>/`:

- `vocals_only.wav` — your converted vocals
- `final_with_music.wav` — converted vocals mixed with the original instrumental

## Credits

- [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) — Zero-shot singing voice synthesis model by Soul AI Lab
