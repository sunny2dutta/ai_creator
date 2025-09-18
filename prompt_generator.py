import os
import openai
from typing import Optional, Tuple
import random
from storage.prompt_storage import PromptStorage, JSONPromptStorage

class FacebookImagePromptGenerator:
    def __init__(self, storage: Optional[PromptStorage] = None):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        self.storage = storage or JSONPromptStorage()

    def generate_prompt(self, profile: str) -> Tuple[str, str]:
        """Generate prompt and caption for a specific profile and randomly selected category"""
        if not self.openai_api_key:
            return self._fallback_prompt(profile)

        # Get profile configuration
        try:
            profile_config = self.storage.get_profile_config(profile)
            categories = self.storage.list_categories(profile)
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

        try:
            # Generate clothing based on hints and category
            clothing_context = ""
            if clothing_hints:
                clothing_context = f"Consider these clothing suggestions: {', '.join(clothing_hints)}"

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

            # Randomly select category using random number generation
            category_index = random.randint(0, len(categories) - 1)
            selected_category = categories[category_index]

            starter = self.storage.get_random_starter_prompt(profile, selected_category)
            return f"{starter}, professional photography, high quality", "Beautiful moment"
        except:
            return "Professional portrait, high quality, perfect lighting", "Beautiful"

if __name__ == "__main__":
    generator = FacebookImagePromptGenerator()
    prompt, caption = generator.generate_prompt("rupashi")
    print(f"Generated prompt: {prompt}")
    print(f"Generated caption: {caption}")