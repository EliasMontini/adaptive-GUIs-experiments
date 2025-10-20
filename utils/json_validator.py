"""
JSON validator and post-processor for ensuring correct structure of steps.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class StepsValidator:
    """Validates and corrects the structure of steps JSON."""

    REQUIRED_ADAPTIVE_FIELDS = [
        "short_text",
        "long_text",
        "image_cobot",
        "image_teach_pendant",
        "video"
    ]

    REQUIRED_STEP_FIELDS = [
        "id",
        "name",
        "category",
        "target_cycle_time",
        "adaptive_fields"
    ]

    @staticmethod
    def validate_step(step: Dict[str, Any]) -> bool:
        """
        Validate if a step has all required fields.

        Args:
            step: Step dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required top-level fields
        for field in StepsValidator.REQUIRED_STEP_FIELDS:
            if field not in step:
                logger.warning(f"Missing field '{field}' in step")
                return False

        # Check adaptive_fields structure
        adaptive_fields = step.get("adaptive_fields", {})
        for field in StepsValidator.REQUIRED_ADAPTIVE_FIELDS:
            if field not in adaptive_fields:
                logger.warning(f"Missing adaptive field '{field}' in step '{step.get('name', 'Unknown')}'")
                return False

        return True

    @staticmethod
    def normalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a step by ensuring all required fields exist and have correct types.

        Args:
            step: Step dictionary to normalize

        Returns:
            Normalized step dictionary
        """
        normalized = {}

        # Ensure id is an integer
        normalized["id"] = int(step.get("id", 0))

        # Ensure name is a string
        normalized["name"] = str(step.get("name", "Unnamed Step")).strip()

        # Ensure category is a string
        normalized["category"] = str(step.get("category", "General")).strip()

        # Ensure target_cycle_time is an integer
        try:
            normalized["target_cycle_time"] = int(step.get("target_cycle_time", 60))
        except (ValueError, TypeError):
            normalized["target_cycle_time"] = 60

        # Ensure adaptive_fields has all required subfields
        adaptive_fields = step.get("adaptive_fields", {})
        if not isinstance(adaptive_fields, dict):
            adaptive_fields = {}

        normalized_adaptive = {}
        for field in StepsValidator.REQUIRED_ADAPTIVE_FIELDS:
            value = adaptive_fields.get(field, "")
            # Ensure all media fields are strings (filenames)
            normalized_adaptive[field] = str(value).strip() if value else ""

        normalized["adaptive_fields"] = normalized_adaptive

        return normalized

    @staticmethod
    def validate_steps_collection(steps: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """
        Validate a collection of steps.

        Args:
            steps: List of step dictionaries

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not isinstance(steps, list):
            errors.append("Steps must be a list")
            return False, errors

        if len(steps) == 0:
            errors.append("Steps list is empty")
            return False, errors

        # Check for duplicate IDs
        ids = [step.get("id") for step in steps if "id" in step]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate step IDs found")

        # Validate each step
        for idx, step in enumerate(steps):
            if not StepsValidator.validate_step(step):
                errors.append(f"Step {idx} has invalid structure")

        return len(errors) == 0, errors

    @staticmethod
    def process_and_normalize(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process and normalize the entire steps data structure.

        Args:
            data: Raw data dictionary from Gemini

        Returns:
            Processed and normalized data, or None if invalid
        """
        if not isinstance(data, dict):
            logger.error("Data must be a dictionary")
            logger.error(f"Received type: {type(data)}")
            return None

        logger.info(f"Raw data keys: {data.keys()}")

        steps = data.get("steps", [])

        if not isinstance(steps, list):
            logger.error("'steps' field must be a list")
            logger.error(f"Received type for 'steps': {type(steps)}")
            logger.error(f"Data content: {data}")
            return None

        if len(steps) == 0:
            logger.error("Steps list is empty")
            logger.error(f"Full data structure: {data}")
            return None

        # Normalize each step
        normalized_steps = []
        for idx, step in enumerate(steps):
            try:
                normalized = StepsValidator.normalize_step(step)
                normalized_steps.append(normalized)
            except Exception as e:
                logger.error(f"Error normalizing step {idx}: {e}")
                return None

        # Ensure IDs are sequential
        for idx, step in enumerate(normalized_steps, start=1):
            step["id"] = idx

        # Validate the final structure
        is_valid, errors = StepsValidator.validate_steps_collection(normalized_steps)

        if not is_valid:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return None

        result = {
            "steps": normalized_steps
        }

        # Preserve any other fields from original data
        for key in data.keys():
            if key != "steps":
                result[key] = data[key]

        logger.info(f"✓ Processed and validated {len(normalized_steps)} steps")
        return result


__all__ = ['StepsValidator']