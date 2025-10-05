#!/usr/bin/env python3
"""
CLI to synthesize sample voices using Hume TTS, optionally preview (play) them,
choose a preferred one, and save it as a named voice in your Hume account.

Prereqs:
- .env with HUME_API_KEY set
- `uv add "hume[microphone]" python-dotenv` already done

Examples:
  # Interactive preview of 2 samples, then save chosen as a named voice
  python -m src.cli.generate_voice \
    --description "Crisp, upper-class British accent with impeccably articulated consonants and perfectly placed vowels. Authoritative and theatrical, as if giving a lecture." \
    --text "The science of speech. That's my profession; also my hobby. Happy is the man who can make a living by his hobby!"

  # Headless (no audio playback, auto choose first) and save wavs
  python -m src.cli.generate_voice \
    --description "Warm, calm narrator with clear enunciation and mild intimacy." \
    --text "Welcome to our journey. Today, we explore the art of empathy in voice." \
    --no-play --auto-select --save-dir ./voices_out
"""

import os
import sys
import asyncio
import base64
import time
import argparse
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.tts import PostedUtterance

# Optional audio playback (available via hume extras)
try:
    from hume.empathic_voice.chat.audio.audio_utilities import play_audio
except Exception:  # pragma: no cover - playback optional in headless envs
    play_audio = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize, preview, and save a Hume TTS voice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--description", required=True, help="Voice style description")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--num-generations", type=int, default=2, help="How many variants to generate")
    parser.add_argument("--name", help="Name for the saved voice (defaults to prefix + timestamp)")
    parser.add_argument("--no-play", action="store_true", help="Do not play audio previews")
    parser.add_argument("--auto-select", action="store_true", help="Pick the first generation automatically")
    parser.add_argument("--save-dir", help="Directory to save raw previews as WAV files")
    return parser.parse_args()


async def synthesize_samples(hume: AsyncHumeClient, description: str, text: str, num_generations: int):
    return await hume.tts.synthesize_json(
        utterances=[
            PostedUtterance(description=description, text=text)
        ],
        num_generations=num_generations,
    )


def b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)


async def maybe_play_audio(audio: bytes, enabled: bool) -> None:
    if not enabled:
        return
    if play_audio is None:
        print("Audio playback not available (missing extras). Skipping preview.")
        return
    await play_audio(audio)


def maybe_save_audio(audio: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(audio)


async def run() -> int:
    load_dotenv()

    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        print("HUME_API_KEY not found in environment variables.")
        return 2

    args = parse_args()

    async with AsyncHumeClient(api_key=api_key) as hume:
        print("Requesting TTS generations…")
        result = await synthesize_samples(hume, args.description, args.text, args.num_generations)

        if not result or not getattr(result, "generations", None):
            print("No generations returned.")
            return 1

        # Preview and optionally save
        chosen_idx: Optional[int] = None
        for idx, generation in enumerate(result.generations, start=1):
            print(f"\nOption {idx} — generation_id: {generation.generation_id}")
            audio_bytes = b64_to_bytes(generation.audio)

            if args.save_dir:
                out_dir = Path(args.save_dir)
                out_path = out_dir / f"preview_{idx}.wav"
                maybe_save_audio(audio_bytes, out_path)
                print(f"Saved preview to {out_path}")

            await maybe_play_audio(audio_bytes, enabled=not args.no_play)

        # Selection
        if args.auto_select:
            chosen_idx = 1
            print("Auto-select enabled — choosing option 1")
        else:
            try:
                user_choice = input(f"\nWhich voice do you prefer? [1-{len(result.generations)}]: ").strip()
            except EOFError:
                user_choice = "1"
                print("No input available, selecting option 1")

            try:
                chosen_idx = int(user_choice)
            except ValueError:
                print("Invalid input, defaulting to 1")
                chosen_idx = 1

        if not (1 <= chosen_idx <= len(result.generations)):
            print("Invalid choice index.")
            return 1

        chosen = result.generations[chosen_idx - 1]
        voice_name = args.name or f"voice-{int(time.time() * 1000)}"

        print(f"\nSaving chosen voice as: {voice_name} (generation_id={chosen.generation_id})")
        await hume.tts.voices.create(name=voice_name, generation_id=chosen.generation_id)
        print(f"Created voice: {voice_name}")

    return 0


def main() -> None:
    try:
        rc = asyncio.run(run())
        sys.exit(rc)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
