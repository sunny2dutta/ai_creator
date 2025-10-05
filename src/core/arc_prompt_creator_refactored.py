"""
Arc Prompt Creator - Production Ready

Transforms simple story concepts into detailed 7-day narrative arcs for social media content.
Uses LLM (GPT-4) to generate authentic, day-by-day story progressions with fallback templates.

Design Choices:
- LLM-first approach with robust fallback ensures reliability
- Interactive editing workflow allows human oversight and refinement
- Langfuse integration for observability and prompt engineering iteration
- Structured JSON output enables programmatic consumption by downstream systems
- Defensive parsing handles malformed LLM responses gracefully

Use Cases:
- Generate week-long content calendars from single story ideas
- Create cohesive narrative arcs for influencer personas
- Automate story planning while maintaining creative control

Author: AI Creator Team
License: MIT
"""

import json
import os
import openai
from typing import Dict, List, Any, Optional
import logging
import tempfile
import subprocess
from dotenv import load_dotenv
import datetime
import argparse
from langfuse import Langfuse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArcPromptCreator:
    """Transforms simple story arcs into detailed 7-day progressions.
    
    Design Choice: Stateless class with dependency injection pattern.
    All configuration comes from environment variables for 12-factor app compliance.
    
    Attributes:
        openai_api_key: OpenAI API key for LLM-enhanced generation
        langfuse_client: Optional Langfuse client for tracing and observability
    """

    def __init__(self):
        """Initialize creator with API clients.
        
        Design Choice: Fail gracefully if optional services unavailable.
        Core functionality (fallback mode) works without any external dependencies.
        """
        # Initialize OpenAI
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("OpenAI API initialized for LLM-enhanced arc generation")
        else:
            logger.warning("OpenAI API key not found. Will use fallback templates.")
        
        # Initialize Langfuse for observability (optional)
        self.langfuse_client = None
        if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
            try:
                self.langfuse_client = Langfuse(
                    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
                    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
                    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                )
                logger.info("Langfuse tracing initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse: {e}")

    def create_7_day_arc(
        self, 
        story_arc: str, 
        character_profile: str = "default"
    ) -> Dict[str, Any]:
        """Break down a simple story arc into 7 days of detailed progression.
        
        Design Choice: Two-tier generation strategy:
        1. LLM-enhanced (GPT-4) for rich, contextual narratives (preferred)
        2. Template-based fallback for reliability when LLM unavailable
        
        Args:
            story_arc: Simple story description (e.g., "Visiting New York", "Beach Holiday")
            character_profile: Character name for consistency across content
            
        Returns:
            Dictionary with 7-day story structure:
            {
                "story_arc": str,
                "character_profile": str,
                "created_at": str (ISO timestamp),
                "data_source": "api" | "fallback",
                "days": {
                    "day_1": {
                        "story": str,
                        "location": str,
                        "activity": str,
                        "mood": str,
                        "rough_prompt": str
                    },
                    ... (day_2 through day_7)
                }
            }
            
        Example:
            creator = ArcPromptCreator()
            arc = creator.create_7_day_arc("NYC Adventure", "traveler_jane")
        """
        logger.info(f"Creating 7-day arc for '{story_arc}' with character '{character_profile}'")
        
        # Create trace for observability
        trace_id = None
        if self.langfuse_client:
            trace_id = self.langfuse_client.create_trace_id()

        # Try LLM-enhanced generation first
        if not self.openai_api_key:
            logger.info("Using fallback template (no OpenAI API key)")
            fallback_data = self._fallback_story_breakdown(story_arc, character_profile)
            fallback_data["data_source"] = "fallback"
            return fallback_data

        try:
            # Create generation span for tracing
            generation = None
            if self.langfuse_client:
                generation = self.langfuse_client.start_generation(
                    name="openai_7day_arc",
                    model="gpt-4o",
                    input={
                        "story_arc": story_arc,
                        "character_profile": character_profile
                    }
                )

            # Call OpenAI API
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
                temperature=0.7  # Balance creativity with consistency
            )

            # Extract response
            ai_response = response.choices[0].message.content.strip()
            
            # Update tracing with response
            if generation:
                generation.update(
                    output=ai_response,
                    usage_details={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )
                generation.end()
            
            # Parse AI response into structured JSON
            story_data = self._parse_ai_response_to_json(
                ai_response, 
                story_arc, 
                character_profile
            )
            story_data["data_source"] = "api"
            
            logger.info(f"Successfully generated 7-day arc using LLM")
            return story_data

        except Exception as e:
            logger.error(f"Error creating 7-day arc with LLM: {e}", exc_info=True)
            logger.info("Falling back to template-based generation")
            
            fallback_data = self._fallback_story_breakdown(story_arc, character_profile)
            fallback_data["data_source"] = "fallback"
            return fallback_data

    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM story breakdown.
        
        Design Choice: Detailed instructions ensure consistent, high-quality output.
        Emphasizes authenticity and social media suitability.
        
        Returns:
            System prompt string
        """
        return """You are an expert storyteller and social media content creator. Your job is to break down a simple story arc into a detailed 7-day progression suitable for social media posts.

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

Make the progression feel natural and authentic, like a real person's journey. Avoid overly dramatic elements - focus on genuine, relatable moments that would work well for social media. Ensure each day builds on the previous one to create a cohesive narrative arc."""

    def _parse_ai_response_to_json(
        self, 
        ai_response: str, 
        story_arc: str, 
        character_profile: str
    ) -> Dict[str, Any]:
        """Parse LLM response into structured JSON format.
        
        Design Choice: Defensive parsing with fallback for each day.
        Handles malformed responses gracefully without failing entire operation.
        
        Args:
            ai_response: Raw text response from LLM
            story_arc: Original story arc for context
            character_profile: Character name for fallbacks
            
        Returns:
            Structured story data dictionary
        """
        story_data = {
            "story_arc": story_arc,
            "character_profile": character_profile,
            "created_at": datetime.datetime.now().isoformat(),
            "data_source": "api",
            "days": {}
        }

        # Split response into lines and parse
        lines = ai_response.split('\n')
        current_day = None
        current_day_data = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for day marker (e.g., "Day 1:", "Day 2:", etc.)
            if line.lower().startswith('day '):
                # Save previous day if exists
                if current_day and current_day_data:
                    story_data["days"][f"day_{current_day}"] = current_day_data

                # Start new day
                try:
                    # Extract day number
                    day_parts = line.split()
                    current_day = int(day_parts[1].rstrip(':'))
                    
                    # Initialize day data
                    current_day_data = {
                        "story": line.split(':', 1)[1].strip() if ':' in line else "",
                        "location": "",
                        "activity": "",
                        "mood": "",
                        "rough_prompt": ""
                    }
                except (IndexError, ValueError) as e:
                    logger.warning(f"Failed to parse day marker '{line}': {e}")
                    continue

            elif current_day and ':' in line:
                # Parse field: value pairs
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
        # Design Choice: Ensure all 7 days present even if LLM output incomplete
        for day_num in range(1, 8):
            day_key = f"day_{day_num}"
            if day_key not in story_data["days"]:
                logger.warning(f"Day {day_num} missing from LLM response, using fallback")
                story_data["days"][day_key] = self._create_fallback_day(
                    day_num, 
                    story_arc, 
                    character_profile
                )

        return story_data

    def _create_fallback_day(
        self, 
        day_num: int, 
        story_arc: str, 
        character_profile: str
    ) -> Dict[str, str]:
        """Create a fallback day when LLM parsing fails.
        
        Design Choice: Generic but functional fallback ensures system never fails.
        
        Args:
            day_num: Day number (1-7)
            story_arc: Story arc description
            character_profile: Character name
            
        Returns:
            Day data dictionary with generic content
        """
        return {
            "story": f"Day {day_num} of {story_arc} - continuing the journey",
            "location": "Various locations during the experience",
            "activity": f"Day {day_num} activities related to {story_arc}",
            "mood": "engaged and present",
            "rough_prompt": f"{character_profile} during day {day_num} of {story_arc}, natural expression"
        }

    def _fallback_story_breakdown(
        self, 
        story_arc: str, 
        character_profile: str
    ) -> Dict[str, Any]:
        """Generate fallback story breakdown when LLM unavailable.
        
        Design Choice: Template-based generation ensures system works offline.
        Useful for development, testing, and cost optimization.
        
        Args:
            story_arc: Story arc description
            character_profile: Character name
            
        Returns:
            Complete 7-day story structure using templates
        """
        logger.info("Generating fallback 7-day arc using templates")
        
        return {
            "story_arc": story_arc,
            "character_profile": character_profile,
            "created_at": datetime.datetime.now().isoformat(),
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

    def save_story_arc(
        self, 
        story_data: Dict[str, Any], 
        profile_name: Optional[str] = None, 
        output_file: Optional[str] = None
    ) -> str:
        """Save story arc data to JSON file.
        
        Design Choice: Flexible output path with sensible defaults.
        Supports both profile-based naming and custom paths.
        
        Args:
            story_data: Story arc data to save
            profile_name: Profile name for default file naming
            output_file: Custom output file path (overrides profile_name)
            
        Returns:
            Path to saved file
            
        Raises:
            IOError: If file cannot be written
        """
        # Determine output file path
        if output_file is None:
            if profile_name:
                # Use profile-based path in data/arcs directory
                arcs_dir = "/Users/debaryadutta/ai_creator/data/arcs"
                os.makedirs(arcs_dir, exist_ok=True)
                output_file = os.path.join(arcs_dir, f"{profile_name}_7day_arc.json")
            else:
                # Default to current directory
                output_file = "7day_arc.json"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved story arc to: {output_file}")
            return output_file
            
        except IOError as e:
            logger.error(f"Failed to save story arc to {output_file}: {e}")
            raise

    def preview_story_arc(self, story_data: Dict[str, Any]) -> None:
        """Print a formatted preview of the story arc.
        
        Design Choice: Human-readable console output for review workflow.
        
        Args:
            story_data: Story arc data to preview
        """
        data_source_label = "🤖 AI Generated" if story_data.get('data_source') == 'api' else "📋 Fallback Template"
        
        print(f"\n{'='*60}")
        print(f"7-DAY STORY ARC: {story_data['story_arc']}")
        print(f"{'='*60}")
        print(f"Character: {story_data['character_profile']}")
        print(f"Data Source: {data_source_label}")
        print(f"Created: {story_data.get('created_at', 'N/A')}")

        for day_key in sorted(story_data['days'].keys()):
            day_data = story_data['days'][day_key]
            day_num = day_key.split('_')[1]

            print(f"\n{'-'*60}")
            print(f"DAY {day_num}")
            print(f"{'-'*60}")
            print(f"📖 Story: {day_data['story']}")
            print(f"📍 Location: {day_data['location']}")
            print(f"🎯 Activity: {day_data['activity']}")
            print(f"😊 Mood: {day_data['mood']}")
            print(f"🎨 Prompt: {day_data['rough_prompt']}")

        print(f"\n{'='*60}\n")

    def confirm_and_edit_story_arc(self, story_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Interactive workflow for confirming or editing generated story arc.
        
        Design Choice: Human-in-the-loop ensures quality control.
        Allows regeneration or manual editing before final save.
        
        Args:
            story_data: Story arc data to review
            
        Returns:
            Confirmed/edited story data, or None if cancelled
        """
        while True:
            self.preview_story_arc(story_data)

            print(f"{'='*60}")
            print("🤔 What would you like to do?")
            print("1. Accept and save the story arc")
            print("2. Edit the entire JSON in your editor")
            print("3. Regenerate the story arc")
            print("4. Cancel")
            print(f"{'='*60}")

            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                logger.info("Story arc accepted")
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
                logger.info("Story arc creation cancelled by user")
                print("❌ Story arc creation cancelled.")
                return None
                
            else:
                print("❌ Invalid choice. Please try again.")

    def _edit_json_in_editor(self, story_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Open JSON in system editor for manual editing.
        
        Design Choice: Uses system default editor for familiar UX.
        Validates JSON after editing to prevent corruption.
        
        Args:
            story_data: Story data to edit
            
        Returns:
            Edited story data, or None if edit failed/cancelled
        """
        # Create temporary file with current JSON
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            json.dump(story_data, temp_file, indent=2, ensure_ascii=False)
            temp_path = temp_file.name

        try:
            # Open in default editor (or vi as fallback)
            editor = os.getenv('EDITOR', 'vi')
            logger.debug(f"Opening editor: {editor}")
            
            subprocess.run([editor, temp_path], check=True)

            # Read back the edited JSON
            with open(temp_path, 'r', encoding='utf-8') as f:
                edited_data = json.load(f)

            logger.info("JSON edited successfully")
            return edited_data

        except subprocess.CalledProcessError as e:
            logger.error(f"Editor exited with error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON after editing: {e}")
            print(f"❌ Invalid JSON: {e}")
            return None
        except FileNotFoundError as e:
            logger.error(f"Editor not found: {e}")
            print(f"❌ Editor '{editor}' not found. Set EDITOR environment variable.")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def main():
    """Command-line interface for arc prompt creator.
    
    Design Choice: Argparse for standard CLI UX with help text.
    """
    parser = argparse.ArgumentParser(
        description='Create 7-day story arc from simple story description',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Beach Holiday" traveler_jane
  %(prog)s "NYC Adventure" mrbananas --output nyc_arc.json
  %(prog)s "Fitness Journey" athlete_alex
        """
    )
    
    parser.add_argument(
        'text', 
        help='Story arc text/description (e.g., "Beach Holiday", "NYC Visit")'
    )
    parser.add_argument(
        'character', 
        help='Character profile name (e.g., "mrbananas", "rupashi")'
    )
    parser.add_argument(
        '--output', 
        help='Output file path (optional, defaults to data/arcs/<character>_7day_arc.json)'
    )
    
    args = parser.parse_args()

    # Create arc prompt creator
    creator = ArcPromptCreator()

    # Generate story arc
    logger.info(f"Generating 7-day arc for '{args.text}' with character '{args.character}'")
    story_data = creator.create_7_day_arc(args.text, args.character)

    # Interactive confirmation and editing
    confirmed_data = creator.confirm_and_edit_story_arc(story_data)
    
    if confirmed_data:
        profile_name = confirmed_data.get('character_profile', args.character)
        output_file = creator.save_story_arc(
            confirmed_data, 
            profile_name=profile_name, 
            output_file=args.output
        )
        print(f"\n✅ Created 7-day arc for: {args.text}")
        print(f"💾 Saved to: {output_file}")
    else:
        print("❌ Story arc creation cancelled.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
