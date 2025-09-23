import json
import datetime
import os
import openai
from typing import Dict, Any
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self, seven_day_arc_path: str = "7day_arc.json"):
        self.seven_day_arc_path = seven_day_arc_path
        self.seven_day_data = self._load_seven_day_arc()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def _load_seven_day_arc(self) -> Dict[str, Any]:
        """Load the 7-day arc JSON file."""
        try:
            with open(self.seven_day_arc_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"7-day arc file not found: {self.seven_day_arc_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {self.seven_day_arc_path}")

    def get_current_day_number(self) -> int:
        """Get current day number based on day of week (1-7, Monday=1)."""
        return datetime.datetime.now().weekday() + 1

    def get_day_content(self, day_number: int = None) -> Dict[str, Any]:
        """Get content for specific day or current day if not specified."""
        if day_number is None:
            day_number = self.get_current_day_number()

        if day_number < 1 or day_number > 7:
            raise ValueError("Day number must be between 1 and 7")

        day_key = f"day_{day_number}"
        if day_key not in self.seven_day_data.get("days", {}):
            raise KeyError(f"Day {day_number} not found in 7-day arc")

        return self.seven_day_data["days"][day_key]

    def _get_prompt_enhancement_system_prompt(self) -> str:
        """System prompt for LLM to enhance image generation prompts."""
        return """
        You are an expert at creating detailed, vivid image generation prompts for AI art tools like DALL-E, Midjourney, and Stable Diffusion.

        Given a basic story scenario, you will create:
        1. An enhanced detailed scene description
        2. Key visual elements that should be emphasized
        3. Mood and lighting descriptors
        4. Technical photography/cinematography details
        5. Style and composition guidance
        6. Detailed clothing and jewelry descriptions that match the character and surroundings

        Focus on:
        - Photorealistic, authentic human expressions and poses
        - Proper lighting and atmospheric details
        - Composition that tells a story
        - Cultural and environmental accuracy
        - Emotional authenticity
        - Specific, detailed clothing that fits the setting and character personality
        - Jewelry and accessories that complement the outfit and location
        - Attention to textures, colors, and style coherence

        Your response should be a JSON object with these fields:
        - enhanced_scene: A detailed 2-3 sentence scene description
        - visual_elements: Array of key visual elements to emphasize
        - mood_descriptors: Array of mood/emotion words
        - lighting: Description of lighting conditions
        - composition: Camera angle and framing suggestions
        - style_notes: Additional style guidance
        - clothing_details: Detailed description of outfit including colors, textures, fit, and style
        - jewelry_accessories: Specific jewelry and accessories that complement the scene and character
        - social_media_caption: An engaging caption for social media posts that captures the story and mood
        """

    def generate_detailed_prompt(self, day_number: int = None) -> Dict[str, Any]:
        """Generate detailed prompt JSON for specific day using LLM enhancement."""
        day_content = self.get_day_content(day_number)
        current_day = day_number or self.get_current_day_number()

        base_info = {
            "story_arc": self.seven_day_data.get("story_arc", ""),
            "character_profile": self.seven_day_data.get("character_profile", ""),
            "data_source": self.seven_day_data.get("data_source", "")
        }

        # Create the input for LLM enhancement
        enhancement_input = {
            "story_arc": base_info["story_arc"],
            "character": base_info["character_profile"],
            "day_story": day_content.get("story", ""),
            "location": day_content.get("location", ""),
            "activity": day_content.get("activity", ""),
            "mood": day_content.get("mood", ""),
            "rough_prompt": day_content.get("rough_prompt", "")
        }

        # Get LLM enhanced prompt details
        enhanced_details = self._enhance_with_llm(enhancement_input)

        # Create detailed prompt structure
        detailed_prompt = {
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "source_arc": base_info["story_arc"],
                "character": base_info["character_profile"],
                "day_number": current_day,
                "data_source": base_info["data_source"],
                "enhancement_method": "llm" if self.openai_api_key else "fallback"
            },
            "day_content": {
                "story_title": day_content.get("story", ""),
                "location": day_content.get("location", ""),
                "activity_description": day_content.get("activity", ""),
                "character_mood": day_content.get("mood", ""),
                "base_prompt": day_content.get("rough_prompt", "")
            },
            "enhanced_prompt": enhanced_details,
            "generation_parameters": {
                "style": "photorealistic",
                "quality": "high",
                "aspect_ratio": "16:9",
                "mood_weight": 0.8,
                "setting_weight": 0.7,
                "character_weight": 0.9
            }
        }

        return detailed_prompt

    def _enhance_with_llm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to enhance the prompt with detailed descriptions."""

        if not self.openai_api_key:
            return self._fallback_enhancement(input_data)

        try:
            user_prompt = f"""
            Enhance this story scenario into detailed image generation prompts:

            Story Arc: {input_data['story_arc']}
            Character: {input_data['character']}
            Day's Story: {input_data['day_story']}
            Location: {input_data['location']}
            Activity: {input_data['activity']}
            Character's Mood: {input_data['mood']}
            Basic Prompt: {input_data['rough_prompt']}

            Create enhanced prompt details that would generate compelling, authentic images for social media.
            """

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_prompt_enhancement_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )

            # Parse the JSON response
            try:
                enhanced_details = json.loads(response.choices[0].message.content.strip())
                return enhanced_details
            except json.JSONDecodeError:
                # If JSON parsing fails, extract key info manually
                content = response.choices[0].message.content.strip()
                return self._parse_llm_response(content, input_data)

        except Exception as e:
            logger.error(f"Error enhancing prompt with LLM: {e}")
            return self._fallback_enhancement(input_data)

    def _parse_llm_response(self, content: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response when JSON parsing fails."""
        lines = content.split('\n')

        enhanced_details = {
            "enhanced_scene": "",
            "visual_elements": [],
            "mood_descriptors": [],
            "lighting": "",
            "composition": "",
            "style_notes": "",
            "clothing_details": "",
            "jewelry_accessories": "",
            "social_media_caption": ""
        }

        current_field = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for field markers
            if "enhanced_scene" in line.lower() or "scene" in line.lower():
                current_field = "enhanced_scene"
                if ":" in line:
                    enhanced_details[current_field] = line.split(":", 1)[1].strip()
            elif "visual_elements" in line.lower() or "visual" in line.lower():
                current_field = "visual_elements"
            elif "mood" in line.lower():
                current_field = "mood_descriptors"
            elif "lighting" in line.lower():
                current_field = "lighting"
                if ":" in line:
                    enhanced_details[current_field] = line.split(":", 1)[1].strip()
            elif "composition" in line.lower():
                current_field = "composition"
                if ":" in line:
                    enhanced_details[current_field] = line.split(":", 1)[1].strip()
            elif "style" in line.lower():
                current_field = "style_notes"
                if ":" in line:
                    enhanced_details[current_field] = line.split(":", 1)[1].strip()
            elif "clothing" in line.lower():
                current_field = "clothing_details"
                if ":" in line:
                    enhanced_details[current_field] = line.split(":", 1)[1].strip()
            elif "jewelry" in line.lower() or "accessories" in line.lower():
                current_field = "jewelry_accessories"
                if ":" in line:
                    enhanced_details[current_field] = line.split(":", 1)[1].strip()
            elif "caption" in line.lower():
                current_field = "social_media_caption"
                if ":" in line:
                    enhanced_details[current_field] = line.split(":", 1)[1].strip()
            elif current_field and (line.startswith("-") or line.startswith("•")):
                # List items
                item = line.lstrip("-•").strip()
                if current_field in ["visual_elements", "mood_descriptors"]:
                    enhanced_details[current_field].append(item)

        # Fill empty fields with meaningful content
        if not enhanced_details["enhanced_scene"]:
            enhanced_details["enhanced_scene"] = f"{input_data['character']} {input_data['activity'].lower()} at {input_data['location']}, appearing {input_data['mood'].lower()}"

        return enhanced_details

    def _fallback_enhancement(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback enhancement when LLM is not available."""
        return {
            "enhanced_scene": f"{input_data['character']} {input_data['activity'].lower()} at {input_data['location']}, appearing {input_data['mood'].lower()}. The scene captures the essence of {input_data['day_story'].lower()}",
            "visual_elements": self._extract_visual_elements_fallback(input_data),
            "mood_descriptors": self._extract_mood_descriptors_fallback(input_data['mood']),
            "lighting": "natural lighting appropriate for the location and time of day",
            "composition": "medium shot focusing on the character with environmental context",
            "style_notes": "photorealistic, authentic expression, social media ready",
            "clothing_details": "outfit appropriate for the setting and character",
            "jewelry_accessories": "accessories that complement the scene and character",
            "social_media_caption": self._generate_fallback_caption(input_data)
        }

    def _generate_fallback_caption(self, input_data: Dict[str, Any]) -> str:
        """Generate a fallback caption for social media."""
        story = input_data.get('day_story', '')
        location = input_data.get('location', '')
        mood = input_data.get('mood', '')

        if story and location:
            return f"{story} at {location}! Feeling {mood.lower()} ✨ #adventure #travel #life"
        elif story:
            return f"{story}! {mood} vibes ✨ #life #moments"
        else:
            return f"Living my best life! {mood} energy ✨ #blessed #life"

    def _extract_visual_elements_fallback(self, input_data: Dict[str, Any]) -> list:
        """Fallback visual element extraction."""
        location = input_data['location'].lower()
        visual_elements = []

        if "times square" in location:
            visual_elements = ["bright neon lights", "bustling crowds", "city energy"]
        elif "central park" in location:
            visual_elements = ["green trees", "peaceful atmosphere", "city skyline"]
        elif "museum" in location:
            visual_elements = ["art galleries", "cultural artifacts", "elegant interiors"]
        elif "broadway" in location:
            visual_elements = ["theater marquees", "evening lights", "entertainment district"]
        elif "brooklyn bridge" in location:
            visual_elements = ["iconic bridge architecture", "city skyline", "urban landscape"]
        elif "empire state building" in location:
            visual_elements = ["panoramic city views", "observation deck", "towering heights"]
        elif "café" in location:
            visual_elements = ["cozy interior", "warm lighting", "intimate setting"]
        else:
            visual_elements = ["authentic environment", "natural setting", "contextual background"]

        return visual_elements

    def _extract_mood_descriptors_fallback(self, mood: str) -> list:
        """Fallback mood descriptor extraction."""
        mood = mood.lower()
        mood_map = {
            "excited": ["energetic", "enthusiastic", "bright-eyed"],
            "overwhelmed": ["wide-eyed", "taking it all in", "slightly stunned"],
            "relaxed": ["peaceful", "serene", "at ease"],
            "content": ["satisfied", "happy", "fulfilled"],
            "inspired": ["thoughtful", "engaged", "captivated"],
            "thrilled": ["delighted", "joyful", "exhilarated"],
            "adventurous": ["curious", "bold", "explorative"],
            "awe-struck": ["amazed", "wonderstruck", "impressed"],
            "reflective": ["contemplative", "pensive", "introspective"]
        }

        for key, descriptors in mood_map.items():
            if key in mood:
                return descriptors

        return ["expressive", "natural", "authentic"]

    def save_prompt(self, detailed_prompt: Dict[str, Any], filename: str = None) -> str:
        """Save the generated detailed prompt to a JSON file."""
        if filename is None:
            filename = "generated_prompt.json"

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(detailed_prompt, file, indent=2, ensure_ascii=False)

        logger.info(f"Saved prompt to: {filename}")
        return filename

    def generate_and_save_today_prompt(self) -> str:
        """Generate prompt for today and save it."""
        detailed_prompt = self.generate_detailed_prompt()
        filename = self.save_prompt(detailed_prompt)
        return filename

if __name__ == "__main__":
    generator = PromptGenerator()
    today_prompt = generator.generate_detailed_prompt()
    saved_file = generator.save_prompt(today_prompt)
    print(f"Generated and saved prompt for day {today_prompt['metadata']['day_number']}: {saved_file}")