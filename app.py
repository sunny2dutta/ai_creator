#!/usr/bin/env python3
"""
Minimal Hume TTS sample aligned with your example:
- Loads .env for HUME_API_KEY
- Synthesizes 2 generations for a single PostedUtterance
- Plays each option
- Prompts for selection
- Saves the selected voice with a timestamped name

You can expand this file with more features later.
"""

import os
import time
import base64
import asyncio
from typing import Optional

from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.tts import PostedUtterance

# Optional playback util (requires hume extras installed)
try:
    from hume.empathic_voice.chat.audio.audio_utilities import play_audio
except Exception:  # Fallback if not available
    play_audio = None


DESCRIPTION = (
    "Crisp, upper-class British accent with impeccably articulated consonants and perfectly "
    "placed vowels. Authoritative and theatrical, as if giving a lecture."
)
TEXT = (
    "The science of speech. That's my profession; also my hobby. Happy is the man who can make "
    "a living by his hobby!"
)


async def synthesize_and_choose(hume: AsyncHumeClient,
                                description: str,
                                text: str,
                                num_generations: int = 2) -> Optional[str]:
    """Synthesize TTS samples, preview them, and return chosen generation_id."""
    result = await hume.tts.synthesize_json(
        utterances=[PostedUtterance(description=description, text=text)],
        num_generations=num_generations,
    )

    if not result or not getattr(result, "generations", None):
        print("No generations returned.")
        return None

    # Preview
    for idx, generation in enumerate(result.generations, start=1):
        print(f"Playing option {idx}…")
        audio_data = base64.b64decode(generation.audio)
        if play_audio is not None:
            await play_audio(audio_data)
        else:
            print("(Audio playback unavailable — install hume extras to enable preview)")

    # Selection
    print("\nWhich voice did you prefer?")
    for idx, generation in enumerate(result.generations, start=1):
        print(f"{idx}. Option {idx} (generation ID: {generation.generation_id})")

    try:
        user_choice = input("Enter your choice (1 or 2): ").strip()
    except EOFError:
        user_choice = "1"
        print("No input available, selecting option 1")

    try:
        selected_index = int(user_choice) - 1
    except ValueError:
        print("Invalid input, selecting option 1")
        selected_index = 0

    if selected_index not in [0, 1]:
        print("Invalid choice. Please select 1 or 2.")
        return None

    chosen = result.generations[selected_index]
    print(f"Selected voice option {selected_index + 1} (generation ID: {chosen.generation_id})")
    return chosen.generation_id


async def run() -> int:
    load_dotenv()

    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise EnvironmentError("HUME_API_KEY not found in environment variables.")

    async with AsyncHumeClient(api_key=api_key) as hume:
        gen_id = await synthesize_and_choose(hume, DESCRIPTION, TEXT, num_generations=2)
        if not gen_id:
            return 1

        voice_name = f"higgins-{int(time.time() * 1000)}"
        await hume.tts.voices.create(name=voice_name, generation_id=gen_id)
        print(f"Created voice: {voice_name}")

    return 0


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
