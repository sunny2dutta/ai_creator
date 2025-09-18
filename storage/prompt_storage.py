from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import os
import random
from pathlib import Path

class PromptStorage(ABC):
    """Abstract base class for prompt storage implementations"""

    @abstractmethod
    def get_starter_prompts(self, profile: str, category: str) -> List[str]:
        """Get starter prompts for a specific profile and category"""
        pass

    @abstractmethod
    def get_clothing_hints(self, profile: str, category: str) -> List[str]:
        """Get clothing hints for a specific profile and category"""
        pass

    @abstractmethod
    def get_profile_config(self, profile: str) -> Dict[str, Any]:
        """Get complete configuration for a profile"""
        pass

    @abstractmethod
    def get_random_starter_prompt(self, profile: str, category: str) -> str:
        """Get a random starter prompt for a category"""
        pass

    @abstractmethod
    def list_profiles(self) -> List[str]:
        """List all available profiles"""
        pass

    @abstractmethod
    def list_categories(self, profile: str) -> List[str]:
        """List all categories for a profile"""
        pass


class JSONPromptStorage(PromptStorage):
    """JSON-based implementation of prompt storage"""

    def __init__(self, data_dir: str = "data/prompts"):
        self.data_dir = Path(data_dir)
        self.profiles_file = Path("data/profiles.json")
        self._cache = {}
        self._load_profiles_index()

    def _load_profiles_index(self):
        """Load the profiles index file"""
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r') as f:
                self.profiles_index = json.load(f)
        else:
            self.profiles_index = {"profiles": {}}

    def _load_profile_data(self, profile: str) -> Dict[str, Any]:
        """Load and cache profile data"""
        if profile in self._cache:
            return self._cache[profile]

        if profile not in self.profiles_index["profiles"]:
            raise ValueError(f"Profile '{profile}' not found in profiles index")

        prompt_file = self.profiles_index["profiles"][profile]["prompt_file"]
        file_path = self.data_dir / prompt_file

        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        self._cache[profile] = data
        return data

    def get_starter_prompts(self, profile: str, category: str) -> List[str]:
        """Get starter prompts for a specific profile and category"""
        data = self._load_profile_data(profile)

        if category not in data["categories"]:
            raise ValueError(f"Category '{category}' not found for profile '{profile}'")

        return data["categories"][category].get("starter_prompts", [])

    def get_clothing_hints(self, profile: str, category: str) -> List[str]:
        """Get clothing hints for a specific profile and category"""
        data = self._load_profile_data(profile)

        if category not in data["categories"]:
            raise ValueError(f"Category '{category}' not found for profile '{profile}'")

        return data["categories"][category].get("clothing_hints", [])

    def get_profile_config(self, profile: str) -> Dict[str, Any]:
        """Get complete configuration for a profile"""
        return self._load_profile_data(profile)

    def get_random_starter_prompt(self, profile: str, category: str) -> str:
        """Get a random starter prompt for a category"""
        prompts = self.get_starter_prompts(profile, category)
        if not prompts:
            return f"Professional photograph of {self.get_profile_config(profile)['topic']}"
        return random.choice(prompts)

    def list_profiles(self) -> List[str]:
        """List all available profiles"""
        return [
            name for name, config in self.profiles_index["profiles"].items()
            if config.get("active", True)
        ]

    def list_categories(self, profile: str) -> List[str]:
        """List all categories for a profile"""
        data = self._load_profile_data(profile)
        return list(data["categories"].keys())

    def get_base_images(self, profile: str) -> List[str]:
        """Get base images for a profile"""
        data = self._load_profile_data(profile)
        return data.get("base_images", [])

    def get_style_preferences(self, profile: str) -> Dict[str, str]:
        """Get style preferences for a profile"""
        data = self._load_profile_data(profile)
        return data.get("style_preferences", {})