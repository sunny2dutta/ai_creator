"""
Google Cloud Storage Uploader

Handles image and video uploads to Google Cloud Storage with automatic retry logic.
Provides public URLs for uploaded media suitable for social media APIs.

Design Choices:
- Separate class for cloud storage operations (Single Responsibility)
- Credentials via environment variable or explicit path (12-factor app)
- UUID-based filenames prevent collisions
- Automatic content type detection
- Comprehensive error handling with retries

Author: AI Creator Team
License: MIT
"""

import io
import uuid
import logging
from typing import Optional, Union
from pathlib import Path
from PIL import Image
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions

from .base_poster import retry_with_backoff

logger = logging.getLogger(__name__)


class GCSUploader:
    """Handles uploads to Google Cloud Storage.
    
    Design Choice: Dedicated class for cloud storage operations.
    Separates storage concerns from posting logic.
    
    Attributes:
        bucket_name: GCS bucket name
        client: Google Cloud Storage client
        bucket: GCS bucket object
    """
    
    def __init__(
        self, 
        bucket_name: str, 
        credentials_path: Optional[str] = None
    ):
        """Initialize GCS uploader.
        
        Design Choice: Credentials path optional - uses GOOGLE_APPLICATION_CREDENTIALS
        env var if not provided (follows Google Cloud best practices).
        
        Args:
            bucket_name: Name of the GCS bucket
            credentials_path: Optional path to service account JSON file
            
        Raises:
            ValueError: If bucket_name is empty
            RuntimeError: If GCS client initialization fails
        """
        if not bucket_name or not bucket_name.strip():
            raise ValueError("bucket_name cannot be empty")
        
        self.bucket_name = bucket_name
        
        try:
            # Initialize GCS client
            if credentials_path:
                self.client = storage.Client.from_service_account_json(credentials_path)
                logger.info(f"GCS client initialized with credentials from: {credentials_path}")
            else:
                # Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
                self.client = storage.Client()
                logger.info("GCS client initialized from environment credentials")
            
            # Get bucket reference
            self.bucket = self.client.bucket(bucket_name)
            
            # Verify bucket exists (fail fast)
            if not self.bucket.exists():
                raise RuntimeError(f"GCS bucket '{bucket_name}' does not exist")
            
            logger.info(f"GCS uploader ready for bucket: {bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS uploader: {e}")
            raise RuntimeError(f"GCS initialization failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def upload_image(
        self, 
        image: Image.Image, 
        prefix: str = "social_media",
        quality: int = 95
    ) -> str:
        """Upload PIL Image to GCS and return public URL.
        
        Design Choice: Accepts PIL Image directly for flexibility.
        Automatic JPEG conversion ensures compatibility.
        
        Args:
            image: PIL Image object to upload
            prefix: Filename prefix for organization
            quality: JPEG quality (1-100, default 95 for high quality)
            
        Returns:
            Public URL of uploaded image
            
        Raises:
            RuntimeError: If upload fails after retries
            
        Example:
            uploader = GCSUploader("my-bucket")
            url = uploader.upload_image(pil_image, prefix="instagram")
        """
        try:
            # Generate unique filename
            filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
            blob = self.bucket.blob(filename)
            
            # Convert PIL Image to JPEG bytes
            img_bytes = io.BytesIO()
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            image.save(img_bytes, format='JPEG', quality=quality, optimize=True)
            img_bytes.seek(0)
            
            # Upload to GCS
            blob.upload_from_file(
                img_bytes, 
                content_type='image/jpeg',
                timeout=60  # 60 second timeout
            )
            
            # Make publicly accessible
            # Design Choice: Public URLs required for social media APIs
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"Uploaded image to GCS: {filename} ({len(img_bytes.getvalue())} bytes)")
            
            return public_url
            
        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"GCS API error during upload: {e}")
            raise RuntimeError(f"Failed to upload image to GCS: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during image upload: {e}")
            raise RuntimeError(f"Image upload failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def upload_video(
        self, 
        video_content: bytes, 
        prefix: str = "social_media"
    ) -> str:
        """Upload video bytes to GCS and return public URL.
        
        Design Choice: Accepts raw bytes for flexibility with different video sources.
        
        Args:
            video_content: Video file content as bytes
            prefix: Filename prefix for organization
            
        Returns:
            Public URL of uploaded video
            
        Raises:
            RuntimeError: If upload fails after retries
            ValueError: If video_content is empty
        """
        if not video_content:
            raise ValueError("video_content cannot be empty")
        
        try:
            # Generate unique filename
            filename = f"{prefix}_{uuid.uuid4().hex}.mp4"
            blob = self.bucket.blob(filename)
            
            # Upload video
            blob.upload_from_string(
                video_content,
                content_type='video/mp4',
                timeout=120  # 2 minute timeout for larger videos
            )
            
            # Make publicly accessible
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"Uploaded video to GCS: {filename} ({len(video_content)} bytes)")
            
            return public_url
            
        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"GCS API error during video upload: {e}")
            raise RuntimeError(f"Failed to upload video to GCS: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during video upload: {e}")
            raise RuntimeError(f"Video upload failed: {str(e)}")
    
    @retry_with_backoff(max_retries=2, base_delay=1)
    def upload_from_path(
        self, 
        file_path: Union[str, Path], 
        prefix: str = "social_media"
    ) -> str:
        """Upload file from local path to GCS.
        
        Design Choice: Convenience method for file-based uploads.
        Auto-detects content type from file extension.
        
        Args:
            file_path: Path to local file
            prefix: Filename prefix for organization
            
        Returns:
            Public URL of uploaded file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If upload fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Determine content type from extension
            extension = file_path.suffix.lower()
            content_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.mp4': 'video/mp4',
                '.mov': 'video/quicktime',
            }
            content_type = content_type_map.get(extension, 'application/octet-stream')
            
            # Generate filename
            filename = f"{prefix}_{uuid.uuid4().hex}{extension}"
            blob = self.bucket.blob(filename)
            
            # Upload file
            blob.upload_from_filename(
                str(file_path),
                content_type=content_type,
                timeout=120
            )
            
            # Make publicly accessible
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"Uploaded file to GCS: {filename} ({file_path.stat().st_size} bytes)")
            
            return public_url
            
        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"GCS API error during file upload: {e}")
            raise RuntimeError(f"Failed to upload file to GCS: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during file upload: {e}")
            raise RuntimeError(f"File upload failed: {str(e)}")
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file from GCS bucket.
        
        Design Choice: Cleanup method for temporary files.
        
        Args:
            filename: Name of file to delete (not full URL)
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            blob = self.bucket.blob(filename)
            blob.delete()
            logger.info(f"Deleted file from GCS: {filename}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete file {filename}: {e}")
            return False
