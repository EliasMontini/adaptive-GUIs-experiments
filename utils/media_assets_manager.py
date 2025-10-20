"""
Media Assets Manager for handling images and videos cataloging and linking.

This module manages the association of media assets (images and videos) with
documentation steps, providing a structured way to store and retrieve media metadata.
"""

from typing import Dict, List, Optional, Any
import json
import os
import logging

logger = logging.getLogger(__name__)


class MediaAsset:
    """Represents a single media asset (image or video)."""

    TYPES = ["image", "video"]
    CATEGORIES = ["teach_pendant", "cobot_action", "assembly"]

    def __init__(
            self,
            filename: str,
            asset_type: str,
            category: str,
            description: str,
            base_path: str
    ):
        """
        Initialize a media asset.

        Args:
            filename: Name of the file (e.g., 'teach_pendant_01.png')
            asset_type: Type of asset ('image' or 'video')
            category: Category of asset ('teach_pendant', 'cobot_action', 'assembly')
            description: Description of what the asset shows
            base_path: Base path where the asset is stored
        """
        if asset_type not in self.TYPES:
            raise ValueError(f"Invalid asset_type. Must be one of {self.TYPES}")
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Must be one of {self.CATEGORIES}")

        self.filename = filename
        self.asset_type = asset_type
        self.category = category
        self.description = description
        self.base_path = base_path

    @property
    def url(self) -> str:
        """Generate the URL path for this asset."""
        return f"{self.base_path}/{self.filename}"

    def to_dict(self) -> Dict[str, str]:
        """Convert asset to dictionary representation."""
        return {
            "url": self.url,
            "type": self.asset_type,
            "category": self.category,
            "description": self.description
        }


class MediaAssetsLibrary:
    """Manages a library of media assets organized by category."""

    def __init__(self, base_media_path: str = ""):
        """
        Initialize the media assets library.

        Args:
            base_media_path: Base path where all media folders are located
        """
        self.base_media_path = base_media_path
        self.assets: Dict[str, List[MediaAsset]] = {
            "teach_pendant": [],
            "cobot_action": [],
            "assembly": []
        }
        self.assets_by_name: Dict[str, MediaAsset] = {}

    def add_image(
            self,
            filename: str,
            category: str,
            description: str
    ) -> MediaAsset:
        """
        Add an image asset to the library.

        Args:
            filename: Name of the image file
            category: Category ('teach_pendant', 'cobot_action', 'assembly')
            description: Description of the image

        Returns:
            Created MediaAsset object
        """
        if category == "teach_pendant":
            base_path = f"{self.base_media_path}/images_cobot"
        elif category == "cobot_action":
            base_path = f"{self.base_media_path}/images_teach_pendant"
        elif category == "assembly":
            base_path = f"{self.base_media_path}/images_assembly"
        else:
            raise ValueError(f"Unknown category: {category}")

        asset = MediaAsset(filename, "image", category, description, base_path)
        self.assets[category].append(asset)
        self.assets_by_name[filename] = asset

        logger.info(f"Added image: {filename} ({category})")
        return asset

    def add_video(
            self,
            filename: str,
            description: str
    ) -> MediaAsset:
        """
        Add a video asset to the library.

        Args:
            filename: Name of the video file
            description: Description of the video

        Returns:
            Created MediaAsset object
        """
        base_path = f"{self.base_media_path}/videos"
        asset = MediaAsset(filename, "video", "assembly", description, base_path)
        self.assets["assembly"].append(asset)
        self.assets_by_name[filename] = asset

        logger.info(f"Added video: {filename}")
        return asset

    def get_asset_by_name(self, filename: str) -> Optional[MediaAsset]:
        """
        Retrieve an asset by filename.

        Args:
            filename: Name of the file to find

        Returns:
            MediaAsset if found, None otherwise
        """
        return self.assets_by_name.get(filename)

    def get_assets_by_category(self, category: str) -> List[MediaAsset]:
        """
        Get all assets in a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of MediaAsset objects in the category
        """
        return self.assets.get(category, [])

    def get_all_assets(self) -> List[MediaAsset]:
        """Get all assets in the library."""
        all_assets = []
        for assets_list in self.assets.values():
            all_assets.extend(assets_list)
        return all_assets

    def to_dict(self) -> Dict[str, Any]:
        """Export the library as a dictionary."""
        return {
            category: [asset.to_dict() for asset in assets]
            for category, assets in self.assets.items()
        }

    def save_to_json(self, filepath: str) -> bool:
        """
        Save the library to a JSON file.

        Args:
            filepath: Path where to save the JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Media library saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save media library: {e}")
            return False

    @classmethod
    def load_from_json(cls, filepath: str, base_media_path: str = "") -> Optional['MediaAssetsLibrary']:
        """
        Load a media library from a JSON file.

        Args:
            filepath: Path to the JSON file
            base_media_path: Base path where media folders are located

        Returns:
            MediaAssetsLibrary instance if successful, None otherwise
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            library = cls(base_media_path)

            # Reconstruct assets from JSON
            for category, assets_data in data.items():
                for asset_data in assets_data:
                    if asset_data["type"] == "image":
                        library.add_image(
                            filename=asset_data["url"].split("/")[-1],
                            category=asset_data["category"],
                            description=asset_data["description"]
                        )
                    elif asset_data["type"] == "video":
                        library.add_video(
                            filename=asset_data["url"].split("/")[-1],
                            description=asset_data["description"]
                        )

            logger.info(f"✓ Media library loaded from {filepath}")
            return library
        except Exception as e:
            logger.error(f"✗ Failed to load media library: {e}")
            return None


class MediaMatcher:
    """Matches media assets to documentation steps based on relevance."""

    @staticmethod
    def find_matching_media(
            step_name: str,
            step_description: str,
            media_library: MediaAssetsLibrary
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Find media assets that match a documentation step.

        Note: This is a basic matching function. You should manually review
        and assign media to steps in the media_assignments.json file.

        Args:
            step_name: Name of the documentation step
            step_description: Description of the step
            media_library: MediaAssetsLibrary to search in

        Returns:
            Dictionary with 'images' and 'videos' keys containing matching media
        """
        combined_text = f"{step_name} {step_description}".lower()

        matching_images = []
        matching_videos = []

        for asset in media_library.get_all_assets():
            asset_desc_lower = asset.description.lower()

            # Simple keyword matching
            if any(keyword in asset_desc_lower for keyword in combined_text.split()):
                if asset.asset_type == "image":
                    matching_images.append(asset.to_dict())
                elif asset.asset_type == "video":
                    matching_videos.append(asset.to_dict())

        return {
            "images": matching_images,
            "videos": matching_videos
        }


__all__ = [
    'MediaAsset',
    'MediaAssetsLibrary',
    'MediaMatcher'
]