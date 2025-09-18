import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv
from storage.prompt_storage import PromptStorage, JSONPromptStorage

load_dotenv()

@dataclass
class ProfileConfig:
    """Configuration for a social media profile"""
    profile_name: str
    facebook_page_id: str
    instagram_business_id: str
    access_token: str
    description: str
    active: bool = True


class ProfileManager:
    """Manages profile configurations with pluggable storage backend"""

    def __init__(self, storage: Optional[PromptStorage] = None):
        self.storage = storage or JSONPromptStorage()
        self._profile_configs = {}
        self._load_profile_configs()

    def _load_profile_configs(self):
        """Load profile configurations from storage and environment"""
        profile_names = self.storage.list_profiles()

        for profile_name in profile_names:
            try:
                # Get profile metadata from storage
                if hasattr(self.storage, 'profiles_index'):
                    profile_meta = self.storage.profiles_index["profiles"][profile_name]
                else:
                    # Fallback for other storage implementations
                    profile_meta = {
                        "facebook_page_id_env": f"FACEBOOK_PAGE_ID_{profile_name.upper()}",
                        "instagram_business_id_env": f"INSTAGRAM_BUSINESS_ID_{profile_name.upper()}",
                        "access_token_env": f"FACEBOOK_LONG_ACCESS_TOKEN_{profile_name.upper()}",
                        "description": f"Profile for {profile_name}",
                        "active": True
                    }

                # Load credentials from environment
                facebook_page_id = os.getenv(profile_meta["facebook_page_id_env"])
                instagram_business_id = os.getenv(profile_meta["instagram_business_id_env"])
                access_token = os.getenv(profile_meta["access_token_env"])

                if facebook_page_id and instagram_business_id and access_token:
                    self._profile_configs[profile_name] = ProfileConfig(
                        profile_name=profile_name,
                        facebook_page_id=facebook_page_id,
                        instagram_business_id=instagram_business_id,
                        access_token=access_token,
                        description=profile_meta.get("description", f"Profile for {profile_name}"),
                        active=profile_meta.get("active", True)
                    )
                else:
                    print(f"Warning: Missing credentials for profile '{profile_name}'")

            except Exception as e:
                print(f"Error loading profile '{profile_name}': {e}")

    def get_profile(self, profile_name: str) -> ProfileConfig:
        """Get a profile configuration"""
        if profile_name not in self._profile_configs:
            raise ValueError(f"Profile '{profile_name}' not found or not properly configured")
        return self._profile_configs[profile_name]

    def list_profiles(self) -> List[str]:
        """List all available and configured profiles"""
        return list(self._profile_configs.keys())

    def get_profile_prompt_data(self, profile_name: str) -> Dict:
        """Get prompt data for a profile"""
        return self.storage.get_profile_config(profile_name)

    def get_categories(self, profile_name: str) -> List[str]:
        """Get categories for a profile"""
        return self.storage.list_categories(profile_name)

    def get_base_images(self, profile_name: str) -> List[str]:
        """Get base images for a profile"""
        if hasattr(self.storage, 'get_base_images'):
            return self.storage.get_base_images(profile_name)
        return []

    def validate_profile(self, profile_name: str) -> bool:
        """Validate that a profile is properly configured"""
        try:
            config = self.get_profile(profile_name)
            # Check that storage has the profile data
            self.storage.get_profile_config(profile_name)
            return True
        except (ValueError, FileNotFoundError):
            return False

    def get_storage(self) -> PromptStorage:
        """Get the underlying storage instance"""
        return self.storage