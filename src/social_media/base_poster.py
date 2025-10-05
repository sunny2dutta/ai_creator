"""
Base Social Media Poster

Abstract base class defining the interface for all social media platform implementations.
Provides common functionality like retry logic and error handling.

Design Choices:
- ABC ensures consistent interface across platforms
- Dataclass for configuration provides type safety
- PostResult standardizes return values
- Retry decorator eliminates code duplication

Author: AI Creator Team
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from PIL import Image
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """Configuration for a social media platform.
    
    Design Choice: Immutable dataclass for type-safe configuration.
    Separates credentials from business logic.
    
    Attributes:
        access_token: Platform API access token
        page_id: Page/account ID (platform-specific)
        business_account_id: Business account ID (optional, for Instagram)
    """
    access_token: str
    page_id: str
    business_account_id: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate configuration on initialization."""
        if not self.access_token or not self.access_token.strip():
            raise ValueError("access_token cannot be empty")
        if not self.page_id or not self.page_id.strip():
            raise ValueError("page_id cannot be empty")


@dataclass
class PostResult:
    """Result of a social media post operation.
    
    Design Choice: Standardized return type for all posting operations.
    Makes error handling and logging consistent.
    
    Attributes:
        success: Whether the post succeeded
        post_id: Platform-specific post ID (if successful)
        error_message: Error description (if failed)
        metadata: Additional platform-specific data
    """
    success: bool
    post_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def retry_with_backoff(max_retries: int = 3, base_delay: int = 1):
    """Decorator for automatic retry with exponential backoff.
    
    Design Choice: Decorator pattern eliminates retry logic duplication.
    Exponential backoff prevents overwhelming failing services.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3)
        def post_to_api(data):
            return requests.post(url, json=data)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on final attempt
                    if attempt == max_retries - 1:
                        break
                    
                    # Calculate wait time with exponential backoff
                    wait_time = base_delay * (2 ** attempt)
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            
            # All retries exhausted
            logger.error(f"{func.__name__} failed after {max_retries} attempts: {last_exception}")
            raise last_exception
            
        return wrapper
    return decorator


class SocialMediaPoster(ABC):
    """Abstract base class for social media platform implementations.
    
    Design Choice: ABC enforces consistent interface across platforms.
    Template method pattern provides common functionality.
    
    Subclasses must implement:
    - post_image_to_feed()
    - post_image_to_story()
    - _validate_credentials()
    """
    
    def __init__(self, config: PlatformConfig):
        """Initialize poster with platform configuration.
        
        Args:
            config: Platform-specific configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_credentials()
        logger.info(f"{self.__class__.__name__} initialized for page: {config.page_id}")
    
    @abstractmethod
    def post_image_to_feed(self, image_url: str, caption: str = "") -> PostResult:
        """Post image to platform feed/timeline.
        
        Args:
            image_url: Publicly accessible URL of the image
            caption: Post caption/description
            
        Returns:
            PostResult with success status and post ID
            
        Raises:
            RuntimeError: If posting fails after retries
        """
        pass
    
    @abstractmethod
    def post_image_to_story(self, image_url: str) -> PostResult:
        """Post image to platform stories.
        
        Args:
            image_url: Publicly accessible URL of the image
            
        Returns:
            PostResult with success status and story ID
            
        Raises:
            RuntimeError: If posting fails after retries
        """
        pass
    
    @abstractmethod
    def _validate_credentials(self) -> None:
        """Validate platform credentials.
        
        Design Choice: Called during initialization to fail fast.
        Each platform implements its own validation logic.
        
        Raises:
            ValueError: If credentials are invalid
        """
        pass
    
    def get_platform_name(self) -> str:
        """Get the name of this platform.
        
        Returns:
            Platform name (e.g., "Facebook", "Instagram")
        """
        return self.__class__.__name__.replace("Poster", "")
