import json
import datetime
import os
import openai
from typing import Dict, Any, List, Optional
import logging
from dotenv import load_dotenv
from langfuse import Langfuse
from pathlib import Path
import pickle
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()
logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self, profile_name: str = "rupashi", seven_day_arc_path: str = None):
        self.profile_name = profile_name
        if seven_day_arc_path is None:
            # Use profile-specific arc file
            self.seven_day_arc_path = f"/Users/debaryadutta/ai_creator/data/arcs/{profile_name}_7day_arc.json"
        else:
            self.seven_day_arc_path = seven_day_arc_path
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Initialize Langfuse for tracing first
        self.langfuse_client = None
        if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
            try:
                self.langfuse_client = Langfuse(
                    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
                    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
                    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                )
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse: {e}")
        
        # Initialize RAG system
        self.rag_enabled = faiss is not None and SentenceTransformer is not None
        self.profile_rags = {}
        self.embedding_model = None

        if self.rag_enabled:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("RAG system initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}")
                self.rag_enabled = False
        
        # Load the seven day data after Langfuse is initialized
        self.seven_day_data = self._load_seven_day_arc()

    def _load_seven_day_arc(self) -> Dict[str, Any]:
        """Load the 7-day arc JSON file."""
        try:
            with open(self.seven_day_arc_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Log successful load if tracing enabled
            if hasattr(self, 'langfuse_client') and self.langfuse_client:
                logger.info(f"Successfully loaded 7-day arc data with {len(data.get('days', {}))} days")
            
            return data
        except FileNotFoundError:
            error_msg = f"7-day arc file not found: {self.seven_day_arc_path}"
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON in file: {self.seven_day_arc_path}"
            raise ValueError(error_msg)

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
        # Start Langfuse span for detailed prompt generation
        trace_id = None
        if self.langfuse_client:
            trace_id = self.langfuse_client.create_trace_id()
        
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

        # Get LLM enhanced prompt details with tracing
        enhanced_details = self._enhance_with_llm(enhancement_input, trace_id)

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

        # Log completion for tracing
        if self.langfuse_client:
            logger.info(f"Detailed prompt generated for day {current_day} using {detailed_prompt['metadata']['enhancement_method']} method")

        return detailed_prompt

    def _get_profile_rag_index(self, profile_name: str) -> Optional[object]:
        """Get or create FAISS index for profile"""
        if not self.rag_enabled:
            return None

        if profile_name not in self.profile_rags:
            try:
                index_dir = Path(f"data/rag_indices/{profile_name}")
                index_path = index_dir / "profile.index"
                data_path = index_dir / "characteristics.pkl"

                if index_path.exists() and data_path.exists():
                    index = faiss.read_index(str(index_path))
                    with open(data_path, 'rb') as f:
                        characteristics = pickle.load(f)
                else:
                    # Create new index if it doesn't exist
                    index_dir.mkdir(parents=True, exist_ok=True)
                    index = faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 embedding dimension
                    characteristics = []

                self.profile_rags[profile_name] = {
                    'index': index,
                    'characteristics': characteristics,
                    'index_dir': index_dir
                }
                
                logger.info(f"Loaded RAG index for {profile_name} with {len(characteristics)} characteristics")
            except Exception as e:
                logger.warning(f"Failed to load RAG index for {profile_name}: {e}")
                return None

        return self.profile_rags.get(profile_name)

    def _load_profile_characteristics(self, profile_name: str) -> List[str]:
        """Load profile characteristics from profile data"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
            from storage.prompt_storage import JSONPromptStorage
            storage = JSONPromptStorage()
            profile_config = storage.get_profile_config(profile_name)
            
            characteristics = []
            
            # Extract characteristics from profile configuration
            if 'characteristics' in profile_config:
                characteristics.extend(profile_config['characteristics'])
            
            if 'style_preferences' in profile_config:
                for pref in profile_config['style_preferences']:
                    characteristics.append(f"Style preference: {pref}")
            
            if 'personality_traits' in profile_config:
                for trait in profile_config['personality_traits']:
                    characteristics.append(f"Personality trait: {trait}")
            
            if 'interests' in profile_config:
                for interest in profile_config['interests']:
                    characteristics.append(f"Interest: {interest}")
                    
            return characteristics
        except Exception as e:
            logger.warning(f"Failed to load profile characteristics for {profile_name}: {e}")
            return []

    def _initialize_profile_rag(self, profile_name: str):
        """Initialize RAG index with profile characteristics"""
        if not self.rag_enabled:
            return

        rag_data = self._get_profile_rag_index(profile_name)
        if not rag_data:
            return

        # If index is empty, populate it
        if len(rag_data['characteristics']) == 0:
            characteristics = self._load_profile_characteristics(profile_name)
            
            if characteristics:
                try:
                    # Generate embeddings
                    embeddings = self.embedding_model.encode(characteristics, normalize_embeddings=True)
                    
                    # Add to FAISS index
                    rag_data['index'].add(embeddings)
                    
                    # Store characteristics
                    rag_data['characteristics'].extend(characteristics)
                    
                    # Save to disk
                    self._save_rag_index(profile_name)
                    logger.info(f"Initialized RAG for {profile_name} with {len(characteristics)} characteristics")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize RAG for {profile_name}: {e}")

    def _save_rag_index(self, profile_name: str):
        """Save RAG index to disk"""
        rag_data = self.profile_rags.get(profile_name)
        if not rag_data:
            return

        try:
            index_path = rag_data['index_dir'] / "profile.index"
            data_path = rag_data['index_dir'] / "characteristics.pkl"

            faiss.write_index(rag_data['index'], str(index_path))
            with open(data_path, 'wb') as f:
                pickle.dump(rag_data['characteristics'], f)
                
        except Exception as e:
            logger.error(f"Failed to save RAG index for {profile_name}: {e}")

    def _extract_rag_enhancements(self, profile_name: str, query: str, k: int = 3) -> List[str]:
        """Extract relevant characteristics from RAG to enhance prompts"""
        if not self.rag_enabled:
            return []

        rag_data = self._get_profile_rag_index(profile_name)
        if not rag_data or rag_data['index'].ntotal == 0:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            
            # Search for similar characteristics
            scores, indices = rag_data['index'].search(query_embedding, min(k * 2, rag_data['index'].ntotal))
            
            # Extract relevant characteristics with score filtering
            relevant_characteristics = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(rag_data['characteristics']) and score > 0.3:
                    char = rag_data['characteristics'][idx]
                    if char not in relevant_characteristics:
                        relevant_characteristics.append(char)
                        
                if len(relevant_characteristics) >= k:
                    break
                    
            return relevant_characteristics
            
        except Exception as e:
            logger.error(f"Failed to extract RAG enhancements: {e}")
            return []

    def _enhance_with_llm(self, input_data: Dict[str, Any], trace_id=None) -> Dict[str, Any]:
        """Use LLM to enhance the prompt with detailed descriptions."""

        if not self.openai_api_key:
            return self._fallback_enhancement(input_data)
        
        # Initialize RAG for this profile if enabled
        character_profile = input_data.get('character', '')
        if self.rag_enabled and character_profile:
            self._initialize_profile_rag(character_profile)
            
        # Extract RAG enhancements for the story scenario
        rag_enhancements = []
        if self.rag_enabled and character_profile:
            story_context = f"{input_data.get('day_story', '')} {input_data.get('activity', '')} {input_data.get('location', '')}"
            rag_enhancements = self._extract_rag_enhancements(character_profile, story_context, k=3)
            if rag_enhancements:
                logger.info(f"RAG Enhancements for {character_profile}: {rag_enhancements}")

        # Create generation for prompt enhancement
        generation = None
        if self.langfuse_client:
            generation = self.langfuse_client.start_generation(
                name="prompt_enhancement_llm",
                model="gpt-4o", 
                input=input_data
            )

        try:
            # Build user prompt with RAG enhancements
            user_prompt = f"""
            Enhance this story scenario into detailed image generation prompts:

            Story Arc: {input_data['story_arc']}
            Character: {input_data['character']}
            Day's Story: {input_data['day_story']}
            Location: {input_data['location']}
            Activity: {input_data['activity']}
            Character's Mood: {input_data['mood']}
            Basic Prompt: {input_data['rough_prompt']}"""
            
            # Add RAG enhancements if available
            if rag_enhancements:
                user_prompt += f"""
            
            Character Profile Insights (incorporate these traits naturally):
            {chr(10).join(f'- {enhancement}' for enhancement in rag_enhancements)}"""
            
            user_prompt += """

            Create enhanced prompt details that would generate compelling, authentic images for social media, incorporating the character's unique traits and personality.
            """

            messages = [
                {
                    "role": "system",
                    "content": self._get_prompt_enhancement_system_prompt()
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )

            # Parse the JSON response
            ai_response = response.choices[0].message.content.strip()
            
            try:
                enhanced_details = json.loads(ai_response)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract key info manually
                enhanced_details = self._parse_llm_response(ai_response, input_data)
            
            # Update generation with response
            if generation:
                generation.update(
                    output=enhanced_details,
                    usage_details={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )
                generation.end()
            
            return enhanced_details

        except Exception as e:
            logger.error(f"Error enhancing prompt with LLM: {e}")
            
            if generation:
                generation.update(
                    output={"error": str(e)},
                    metadata={"enhancement_failed": True, "error": str(e)}
                )
                generation.end()
            
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
        
        # Try to get RAG enhancements for fallback too
        character_profile = input_data.get('character', '')
        rag_enhancements = []
        if self.rag_enabled and character_profile:
            self._initialize_profile_rag(character_profile)
            story_context = f"{input_data.get('day_story', '')} {input_data.get('activity', '')} {input_data.get('location', '')}"
            rag_enhancements = self._extract_rag_enhancements(character_profile, story_context, k=2)
            if rag_enhancements:
                logger.info(f"Fallback RAG Enhancements for {character_profile}: {rag_enhancements}")
        
        enhanced_scene = f"{input_data['character']} {input_data['activity'].lower()} at {input_data['location']}, appearing {input_data['mood'].lower()}. The scene captures the essence of {input_data['day_story'].lower()}"
        
        # Incorporate RAG enhancements into scene description
        if rag_enhancements:
            enhanced_scene += f", showcasing their {', '.join(rag_enhancements[:2]).lower()}"
        
        return {
            "enhanced_scene": enhanced_scene,
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
            # Use the character profile name to create the output filename
            character = detailed_prompt.get('metadata', {}).get('character', 'profile')
            filename = f"{character}_7day_arc.json"

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