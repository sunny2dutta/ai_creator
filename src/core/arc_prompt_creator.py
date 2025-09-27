"""
Arc Prompt Creator
Takes simple story arcs like "Visiting New York" and breaks them into detailed 7-day story progressions
"""

import json
import os
import openai
from typing import Dict, List, Any
import logging
import tempfile
import subprocess
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()


logger = logging.getLogger(__name__)

class ArcPromptCreator:
    """Breaks down simple story arcs into detailed 7-day progressions"""

    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def create_7_day_arc(self, story_arc: str, character_profile: str = "mrbananas") -> Dict[str, Any]:
        """
        Break down a simple story arc into 7 days of detailed story progression

        Args:
            story_arc: Simple story description like "Visiting New York"
            character_profile: Character name for consistency

        Returns:
            JSON structure with 7 days of story details
        """

        if not self.openai_api_key:
            fallback_data = self._fallback_story_breakdown(story_arc, character_profile)
            fallback_data["data_source"] = "fallback"
            return fallback_data

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"Create a 7-day story progression for: '{story_arc}' featuring character '{character_profile}'"
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )

            # Parse the AI response to extract structured data
            ai_response = response.choices[0].message.content.strip()
            story_data = self._parse_ai_response_to_json(ai_response, story_arc, character_profile)
            story_data["data_source"] = "api"

            return story_data

        except Exception as e:
            logger.error(f"Error creating 7-day arc with AI: {e}")
            fallback_data = self._fallback_story_breakdown(story_arc, character_profile)
            fallback_data["data_source"] = "fallback"
            return fallback_data

    def _get_system_prompt(self) -> str:
        """Get the system prompt for AI story breakdown"""
        return """
        You are an expert storyteller and social media content creator. Your job is to break down a simple story arc into a detailed 7-day progression suitable for social media posts.

        For each day, provide:
        1. Story progression (what happens this day)
        2. Location/setting where this takes place
        3. Main activity or focus for the day
        4. Character's mood/emotion for authentic expression
        5. A rough prompt for image generation

        Structure your response as a clear day-by-day breakdown:

        Day 1: [Story element]
        Location: [Where this happens]
        Activity: [What they're doing]
        Mood: [How they feel]
        Rough Prompt: [Basic image prompt]

        [Continue for all 7 days]

        Make the progression feel natural and authentic, like a real person's journey. Avoid overly dramatic elements - focus on genuine, relatable moments that would work well for social media.
        """

    def _parse_ai_response_to_json(self, ai_response: str, story_arc: str, character_profile: str) -> Dict[str, Any]:
        """Parse AI response into structured JSON format"""

        story_data = {
            "story_arc": story_arc,
            "character_profile": character_profile,
            "created_at": "now",
            "data_source": "api",
            "days": {}
        }

        # Split response into days and parse each
        lines = ai_response.split('\n')
        current_day = None
        current_day_data = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for day marker
            if line.lower().startswith('day '):
                # Save previous day if exists
                if current_day and current_day_data:
                    story_data["days"][f"day_{current_day}"] = current_day_data

                # Start new day
                try:
                    current_day = int(line.split()[1].rstrip(':'))
                    current_day_data = {
                        "story": line.split(':', 1)[1].strip() if ':' in line else "",
                        "location": "",
                        "activity": "",
                        "mood": "",
                        "rough_prompt": ""
                    }
                except (IndexError, ValueError):
                    continue

            elif current_day and ':' in line:
                key_part = line.split(':', 1)[0].lower()
                value_part = line.split(':', 1)[1].strip()

                if 'location' in key_part:
                    current_day_data["location"] = value_part
                elif 'activity' in key_part:
                    current_day_data["activity"] = value_part
                elif 'mood' in key_part:
                    current_day_data["mood"] = value_part
                elif 'prompt' in key_part:
                    current_day_data["rough_prompt"] = value_part

        # Save the last day
        if current_day and current_day_data:
            story_data["days"][f"day_{current_day}"] = current_day_data

        # Fill any missing days with fallback
        for day_num in range(1, 8):
            day_key = f"day_{day_num}"
            if day_key not in story_data["days"]:
                story_data["days"][day_key] = self._create_fallback_day(day_num, story_arc, character_profile)

        return story_data

    def _create_fallback_day(self, day_num: int, story_arc: str, character_profile: str) -> Dict[str, str]:
        """Create a fallback day when AI parsing fails"""
        return {
            "story": f"Day {day_num} of {story_arc} - continuing the journey",
            "location": "Various locations during the experience",
            "activity": f"Day {day_num} activities related to {story_arc}",
            "mood": "engaged and present",
            "rough_prompt": f"{character_profile} during day {day_num} of {story_arc}, natural expression"
        }

    def _fallback_story_breakdown(self, story_arc: str, character_profile: str) -> Dict[str, Any]:
        """Fallback story breakdown when AI is not available"""

        # Create basic progression based on story type
        return self._create_generic_fallback(story_arc, character_profile)

    def _create_generic_fallback(self, story_arc: str, character_profile: str) -> Dict[str, Any]:
        """Create generic 7-day progression"""
        return {
            "story_arc": story_arc,
            "character_profile": character_profile,
            "created_at": "now",
            "data_source": "fallback",
            "days": {
                f"day_{i}": {
                    "story": f"Day {i} of {story_arc} - continuing the journey",
                    "location": f"Various locations relevant to {story_arc}",
                    "activity": f"Day {i} activities related to {story_arc}",
                    "mood": "engaged and present",
                    "rough_prompt": f"{character_profile} on day {i} of {story_arc}, natural expression"
                } for i in range(1, 8)
            }
        }

    def save_story_arc(self, story_data: Dict[str, Any], profile_name: str = None, output_file: str = None) -> str:
        """Save the story arc data to JSON file"""
        if output_file is None:
            if profile_name:
                output_file = f"/Users/debaryadutta/ai_creator/data/arcs/{profile_name}_7day_arc.json"
            else:
                output_file = "7day_arc.json"

        with open(output_file, 'w') as f:
            json.dump(story_data, f, indent=2)

        logger.info(f"Saved story arc to: {output_file}")
        return output_file

    def preview_story_arc(self, story_data: Dict[str, Any]) -> None:
        """Print a preview of the story arc"""
        data_source_label = "🤖 AI Generated" if story_data.get('data_source') == 'api' else "📋 Fallback Template"
        print(f"\n=== 7-DAY STORY ARC: {story_data['story_arc']} ===")
        print(f"Character: {story_data['character_profile']}")
        print(f"Data Source: {data_source_label}")

        for day_key in sorted(story_data['days'].keys()):
            day_data = story_data['days'][day_key]
            day_num = day_key.split('_')[1]

            print(f"\n--- DAY {day_num} ---")
            print(f"Story: {day_data['story']}")
            print(f"Location: {day_data['location']}")
            print(f"Activity: {day_data['activity']}")
            print(f"Mood: {day_data['mood']}")
            print(f"Rough Prompt: {day_data['rough_prompt']}")

    def confirm_and_edit_story_arc(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Allow user to confirm or edit the generated story arc"""
        import tempfile
        import subprocess

        while True:
            self.preview_story_arc(story_data)

            print(f"\n{'='*60}")
            print("🤔 What would you like to do?")
            print("1. Accept and save the story arc")
            print("2. Edit the entire JSON in your editor")
            print("3. Regenerate the story arc")
            print("4. Cancel")

            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                return story_data
            elif choice == "2":
                edited_data = self._edit_json_in_editor(story_data)
                if edited_data:
                    story_data = edited_data
                    print("✅ JSON updated successfully!")
                else:
                    print("❌ Edit cancelled or invalid JSON.")
            elif choice == "3":
                print("\n🔄 Regenerating story arc...")
                return self.create_7_day_arc(
                    story_data['story_arc'],
                    story_data['character_profile']
                )
            elif choice == "4":
                print("❌ Story arc creation cancelled.")
                return None
            else:
                print("❌ Invalid choice. Please try again.")

    def _edit_json_in_editor(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Open JSON in system editor for editing"""

        # Create temporary file with current JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(story_data, temp_file, indent=2)
            temp_path = temp_file.name

        try:
            # Open in default editor (or vi as fallback)
            editor = os.getenv('EDITOR', 'vi')
            subprocess.run([editor, temp_path], check=True)

            # Read back the edited JSON
            with open(temp_path, 'r') as f:
                edited_data = json.load(f)

            return edited_data

        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error editing JSON: {e}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Example usage and testing
if __name__ == "__main__":

    # Test the arc prompt creator
    creator = ArcPromptCreator()

    # Example 1: Travel story
    story_arc = "Beach Holiday"
    story_data = creator.create_7_day_arc(story_arc, "mrbananas")

    # Get user confirmation and allow editing
    confirmed_data = creator.confirm_and_edit_story_arc(story_data)
    if confirmed_data:
        profile_name = confirmed_data.get('character_profile', 'mrbananas')
        output_file = creator.save_story_arc(confirmed_data, profile_name=profile_name)
        print(f"\n✅ Created 7-day arc for: {story_arc}")
        print(f"💾 Saved to: {output_file}")
    else:
        print("❌ Story arc creation cancelled.")


    # Get user confirmation and allow editing
