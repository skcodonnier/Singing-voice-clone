"""
SoulX-Singer SVC Pipeline
Convert any song into your voice using SoulX-Singer.

Usage:
    python convert.py --voice DATA/shantanu.wav --song DATA/Test.wav
    python convert.py --voice DATA/shantanu.wav --song "Ed Sheeran - Perfect.wav" --steps 8
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Resolve project root before any chdir
PROJECT_ROOT = Path(__file__).parent.resolve()
SOULX_DIR = PROJECT_ROOT / "SoulX-Singer"
OUTPUT_DIR = PROJECT_ROOT / "audio_output"

# Add SoulX-Singer to path and chdir (needed for relative model paths)
sys.path.insert(0, str(SOULX_DIR))
os.chdir(str(SOULX_DIR))


def get_output_path(song_path: str) -> Path:
    """Generate unique output path: audio_output/<songname>_<timestamp>/"""
    song_name = Path(song_path).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"{song_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def preprocess_voice(voice_path: str, work_dir: Path, device: str):
    """Preprocess prompt voice: extract F0 (no vocal separation needed)."""
    from preprocess.pipeline import PreprocessPipeline

    print(f"\n[1/4] Preprocessing your voice: {voice_path}")
    t0 = time.time()

    # Trim voice reference to max 30 seconds (model requirement)
    import librosa
    import soundfile as sf
    audio, sr = librosa.load(voice_path, sr=44100, mono=True)
    max_samples = 30 * sr
    if len(audio) > max_samples:
        print(f"  Trimming voice reference to 30s (was {len(audio)/sr:.1f}s)")
        audio = audio[:max_samples]
    trimmed_path = work_dir / "voice_trimmed.wav"
    sf.write(str(trimmed_path), audio, sr)

    pipeline = PreprocessPipeline(
        device=device,
        language="English",
        save_dir=str(work_dir),
        vocal_sep=False,
        midi_transcribe=False,
    )
    pipeline.run(audio_path=str(trimmed_path), vocal_sep=False, language="English")

    print(f"  Done in {time.time() - t0:.1f}s")
    return work_dir / "vocal.wav", work_dir / "vocal_f0.npy"


def preprocess_song(song_path: str, work_dir: Path, device: str):
    """Preprocess target song: separate vocals + extract F0."""
    from preprocess.pipeline import PreprocessPipeline

    print(f"\n[2/4] Preprocessing song (vocal isolation + F0): {song_path}")
    print("  This may take 10-20 minutes on CPU...")
    t0 = time.time()

    pipeline = PreprocessPipeline(
        device=device,
        language="English",
        save_dir=str(work_dir),
        vocal_sep=True,
        midi_transcribe=False,
    )
    pipeline.run(audio_path=song_path, vocal_sep=True, language="English")

    print(f"  Done in {time.time() - t0:.1f}s")
    return (
        work_dir / "vocal.wav",
        work_dir / "vocal_f0.npy",
        work_dir / "acc.wav",
    )


def run_svc(
    prompt_wav: Path,
    prompt_f0: Path,
    target_wav: Path,
    target_f0: Path,
    out_dir: Path,
    device: str,
    n_steps: int,
    cfg: float,
    pitch_shift: int,
):
    """Run SoulX-Singer SVC inference."""
    from cli.inference_svc import build_model, process
    from soulxsinger.utils.file_utils import load_config

    print(f"\n[3/4] Running voice conversion ({n_steps} diffusion steps)...")
    print(f"  Model: 698M params | Device: {device}")
    t0 = time.time()

    config = load_config("soulxsinger/config/soulxsinger.yaml")
    model = build_model(
        model_path="pretrained_models/SoulX-Singer/model-svc.pt",
        config=config,
        device=device,
        use_fp16="cuda" in device,
    )

    class Args:
        pass

    args = Args()
    args.device = device
    args.prompt_wav_path = str(prompt_wav)
    args.target_wav_path = str(target_wav)
    args.prompt_f0_path = str(prompt_f0)
    args.target_f0_path = str(target_f0)
    args.save_dir = str(out_dir)
    args.auto_shift = True
    args.pitch_shift = pitch_shift
    args.n_steps = n_steps
    args.cfg = cfg
    args.use_fp16 = "cuda" in device

    process(args, config, model)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return out_dir / "generated.wav"


def mix_audio(vocals_path: Path, instrumental_path: Path, out_path: Path):
    """Mix converted vocals with original instrumental."""
    import subprocess

    print(f"\n[4/4] Mixing vocals + instrumental...")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(vocals_path),
            "-i", str(instrumental_path),
            "-filter_complex",
            "[0:a]aresample=44100[vocals];"
            "[1:a]aresample=44100[inst];"
            "[vocals]volume=1.0[v];"
            "[inst]volume=0.7[i];"
            "[v][i]amix=inputs=2:duration=longest:normalize=0[out]",
            "-map", "[out]",
            "-ar", "44100", "-ac", "1",
            str(out_path),
        ],
        capture_output=True,
    )
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert a song into your voice")
    parser.add_argument("--voice", required=True, help="Path to your voice recording (WAV)")
    parser.add_argument("--song", required=True, help="Path to the song to convert (WAV/MP3)")
    parser.add_argument("--steps", type=int, default=4, help="Diffusion steps (4=fast, 8=balanced, 32=best quality)")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale (1.0-3.0, higher=more timbre similarity)")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not set)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Resolve paths BEFORE chdir to SoulX-Singer dir
    voice_path = str(Path(args.voice).resolve())
    song_path = str(Path(args.song).resolve())

    print("=" * 60)
    print("  SoulX-Singer Voice Conversion")
    print("=" * 60)
    print(f"  Voice:    {voice_path}")
    print(f"  Song:     {song_path}")
    print(f"  Device:   {args.device}")
    print(f"  Steps:    {args.steps}")
    print(f"  CFG:      {args.cfg}")
    print(f"  Pitch:    {args.pitch}")

    # Create unique output directory
    out_dir = get_output_path(song_path)
    work_dir = out_dir / "workspace"
    print(f"  Output:   {out_dir}")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Preprocess voice
    prompt_work = work_dir / "prompt"
    prompt_work.mkdir(parents=True, exist_ok=True)
    prompt_wav, prompt_f0 = preprocess_voice(voice_path, prompt_work, args.device)

    # Step 2: Preprocess song
    target_work = work_dir / "target"
    target_work.mkdir(parents=True, exist_ok=True)
    target_wav, target_f0, acc_wav = preprocess_song(song_path, target_work, args.device)

    # Step 3: Run SVC
    svc_out = out_dir / "svc_output"
    svc_out.mkdir(parents=True, exist_ok=True)
    generated = run_svc(
        prompt_wav, prompt_f0,
        target_wav, target_f0,
        svc_out, args.device,
        args.steps, args.cfg, args.pitch,
    )

    # Copy vocals-only output
    import shutil
    vocals_only = out_dir / "vocals_only.wav"
    shutil.copy2(str(generated), str(vocals_only))
    print(f"  Vocals only: {vocals_only}")

    # Step 4: Mix with instrumental
    final_output = out_dir / "final_with_music.wav"
    if acc_wav.exists():
        mix_audio(generated, acc_wav, final_output)
    else:
        print("  [!] No instrumental found, skipping mix")

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Time:         {total / 60:.1f} minutes")
    print(f"  Vocals only:  {vocals_only}")
    if final_output.exists():
        print(f"  With music:   {final_output}")
    print(f"  All files:    {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
