import os
import openai
from typing import Optional, Tuple, List, Dict, Any
import random
import logging
from pathlib import Path
import pickle
import numpy as np
from storage.prompt_storage import PromptStorage, JSONPromptStorage

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class FacebookImagePromptGenerator:
    def __init__(self, storage: Optional[PromptStorage] = None):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        self.storage = storage or JSONPromptStorage()

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

    def _get_profile_rag_index(self, profile_name: str) -> Optional[object]:
        """Get or create FAISS index for profile"""
        if not self.rag_enabled:
            return None

        if profile_name not in self.profile_rags:
            try:
                index_dir = Path(f"data/rag_indices/{profile_name}")
                index_dir.mkdir(parents=True, exist_ok=True)

                # Try to load existing index
                index_path = index_dir / "profile.index"
                data_path = index_dir / "characteristics.pkl"

                if index_path.exists() and data_path.exists():
                    index = faiss.read_index(str(index_path))
                    with open(data_path, 'rb') as f:
                        characteristics = pickle.load(f)
                else:
                    # Create new index
                    index = faiss.IndexFlatIP(384)  # dimension for all-MiniLM-L6-v2
                    characteristics = []

                self.profile_rags[profile_name] = {
                    'index': index,
                    'characteristics': characteristics,
                    'index_dir': index_dir
                }

            except Exception as e:
                logger.warning(f"Failed to load RAG index for {profile_name}: {e}")
                return None

        return self.profile_rags.get(profile_name)

    def _initialize_profile_rag(self, profile_name: str, profile_config: Dict[str, Any]):
        """Initialize RAG index with profile characteristics"""
        if not self.rag_enabled:
            return

        rag_data = self._get_profile_rag_index(profile_name)
        if not rag_data:
            return

        # Skip if already initialized (has characteristics)
        if len(rag_data['characteristics']) > 0:
            return

        characteristics_to_add = []

        # Extract characteristics from profile config
        if "description" in profile_config:
            characteristics_to_add.append({
                "content": profile_config["description"],
                "type": "description",
                "category": "general"
            })

        if "topic" in profile_config:
            characteristics_to_add.append({
                "content": f"Content focused on {profile_config['topic']}",
                "type": "topic",
                "category": "general"
            })

        # Style preferences
        if "style_preferences" in profile_config:
            for key, value in profile_config["style_preferences"].items():
                characteristics_to_add.append({
                    "content": f"{key}: {value}",
                    "type": "style",
                    "category": "style"
                })

        # Category data
        if "categories" in profile_config:
            for category, category_data in profile_config["categories"].items():
                if "starter_prompts" in category_data:
                    for prompt in category_data["starter_prompts"][:3]:  # Limit to avoid too many
                        characteristics_to_add.append({
                            "content": prompt,
                            "type": "prompt_template",
                            "category": category
                        })

                if "clothing_hints" in category_data:
                    for hint in category_data["clothing_hints"][:3]:  # Limit to avoid too many
                        characteristics_to_add.append({
                            "content": f"Clothing style: {hint}",
                            "type": "clothing",
                            "category": category
                        })

        # Add to FAISS index
        if characteristics_to_add:
            try:
                contents = [char["content"] for char in characteristics_to_add]
                embeddings = self.embedding_model.encode(contents, normalize_embeddings=True)

                # Add to index
                rag_data['index'].add(embeddings)

                # Store characteristics
                for i, char in enumerate(characteristics_to_add):
                    rag_data['characteristics'].append(char)

                # Save to disk
                self._save_rag_index(profile_name)
                logger.info(f"Initialized RAG for {profile_name} with {len(characteristics_to_add)} characteristics")

            except Exception as e:
                logger.error(f"Failed to add characteristics to RAG: {e}")

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

    def _extract_rag_enhancements(self, profile_name: str, query: str, category: str = None, k: int = 3) -> List[str]:
        """Extract relevant characteristics from RAG to enhance prompts"""
        if not self.rag_enabled:
            return []

        rag_data = self._get_profile_rag_index(profile_name)
        if not rag_data or rag_data['index'].ntotal == 0:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)

            # Search FAISS index
            scores, indices = rag_data['index'].search(query_embedding, min(k * 2, rag_data['index'].ntotal))

            # Extract relevant characteristics
            enhancements = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(rag_data['characteristics']) and score > 0.3:
                    char = rag_data['characteristics'][idx]

                    # Filter by category if specified
                    if category and char["category"] != category and char["category"] != "general":
                        continue

                    # Add relevant characteristics for prompt enhancement
                    if char["type"] in ["style", "clothing", "prompt_template"]:
                        enhancements.append(char["content"])

                    if len(enhancements) >= k:
                        break

            return enhancements

        except Exception as e:
            logger.error(f"Failed to extract RAG enhancements: {e}")
            return []

    def generate_prompt(self, profile: str) -> Tuple[str, str]:
        """Generate prompt and caption for a specific profile and randomly selected category"""
        if not self.openai_api_key:
            return self._fallback_prompt(profile)

        # Get profile configuration
        try:
            profile_config = self.storage.get_profile_config(profile)
            categories = self.storage.list_categories(profile)

            # Initialize RAG if not already done
            if self.rag_enabled:
                self._initialize_profile_rag(profile, profile_config)

        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading profile '{profile}': {e}")
            return self._fallback_prompt(profile)

        # Randomly select category using random number generation
        category_index = random.randint(0, len(categories) - 1)
        selected_category = categories[category_index]

        print(f"Profile: {profile}, Category: {selected_category}")

        # Get starter prompt and clothing hints
        starter_prompts = self.storage.get_starter_prompts(profile, selected_category)
        clothing_hints = self.storage.get_clothing_hints(profile, selected_category)

        base_starter = random.choice(starter_prompts) if starter_prompts else f"Professional photograph of {profile_config['topic']}"

        # Extract RAG enhancements for this prompt and category
        rag_enhancements = []
        if self.rag_enabled:
            rag_enhancements = self._extract_rag_enhancements(profile, base_starter, selected_category, 3)
            if rag_enhancements:
                print(f"RAG Enhancements: {rag_enhancements}")

        try:
            # Generate clothing based on hints and category
            clothing_context = ""
            if clothing_hints:
                clothing_context = f"Consider these clothing suggestions: {', '.join(clothing_hints)}"

            # Add RAG enhancements to clothing context
            if rag_enhancements:
                rag_context = f"Also incorporate these profile characteristics: {', '.join(rag_enhancements)}"
                clothing_context += f" {rag_context}" if clothing_context else rag_context

            clothing = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a professional fashion stylist. "
                            f"What type of outfit (clothes and jewellery) should a {selected_category} woman wear for {profile_config['topic']}? "
                            f"Preferably use bright colors. {clothing_context}"
                        )
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            clothing_choice = clothing.choices[0].message.content.strip()
            print(f"Clothing: {clothing_choice}")

            # Get style preferences
            style_prefs = self.storage.get_style_preferences(profile) if hasattr(self.storage, 'get_style_preferences') else {}

            # Prepare RAG enhancement text for prompt generation
            rag_enhancement_text = ""
            if rag_enhancements:
                rag_enhancement_text = f"- Incorporate these profile characteristics: {', '.join(rag_enhancements)} "

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a professional AI prompt engineer for influencer content. "
                            f"Generate highly detailed, photorealistic prompts for image generation. "
                            f"Base your prompt on this starter: '{base_starter}' "
                            f"Rules:\n"
                            f"- The subject is always the SAME woman (same face, body type, ethnicity, and age). "
                            f"- Portray her as {selected_category} "
                            f"- Outfit should be {clothing_choice} "
                            f"- Style preferences: {style_prefs} "
                            f"{rag_enhancement_text}"
                            f"- Describe: pose, lighting, camera setup, outfit, background, and mood. "
                            f"- Camera realism is critical (e.g., Canon EOS R5, 85mm f/1.2, shallow depth of field). "
                            f"- Important: it should not change anything in appearance other than clothes"
                        )
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )

            caption = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Create a 3 word witty caption for this image: {response.choices[0].message.content.strip()}"
                        )
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )

            # Add technical specifications
            camera_specs = style_prefs.get('camera', 'Canon EOS R5, 85mm f/1.2 RF lens')
            quality = style_prefs.get('quality', '8k photo, ultra-realistic portrait')

            final_prompt = f"{response.choices[0].message.content.strip()}, {camera_specs}, {quality}"

            return final_prompt, caption.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI API failed: {e}")
            return self._fallback_prompt(profile)

    def _fallback_prompt(self, profile: str) -> Tuple[str, str]:
        """Fallback prompt generation when OpenAI is not available"""
        try:
            profile_config = self.storage.get_profile_config(profile)
            categories = self.storage.list_categories(profile)

            # Initialize RAG if not already done
            if self.rag_enabled:
                self._initialize_profile_rag(profile, profile_config)

            # Randomly select category using random number generation
            category_index = random.randint(0, len(categories) - 1)
            selected_category = categories[category_index]

            starter = self.storage.get_random_starter_prompt(profile, selected_category)

            # Try to enhance with RAG
            enhanced_starter = starter
            if self.rag_enabled:
                rag_enhancements = self._extract_rag_enhancements(profile, starter, selected_category, 2)
                if rag_enhancements:
                    enhanced_starter = f"{starter}, incorporating: {', '.join(rag_enhancements)}"
                    print(f"Fallback RAG Enhancements: {rag_enhancements}")

            return f"{enhanced_starter}, professional photography, high quality", "Beautiful moment"
        except:
            return "Professional portrait, high quality, perfect lighting", "Beautiful"

if __name__ == "__main__":
    generator = FacebookImagePromptGenerator()
    prompt, caption = generator.generate_prompt("rupashi")
    print(f"Generated prompt: {prompt}")
    print(f"Generated caption: {caption}")