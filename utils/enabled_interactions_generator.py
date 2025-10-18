"""
Generator for enabled_interactions.json based on completed_steps.json.
Analyzes which media assets are present in each step and creates boolean flags.
"""

from typing import Dict, List, Any, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


class EnabledInteractionsGenerator:
    """Generates enabled interactions configuration based on available media in steps."""

    @staticmethod
    def generate(completed_steps: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate enabled interactions configuration from completed steps.

        Args:
            completed_steps: Dictionary containing the steps with adaptive_fields

        Returns:
            Dictionary with enabled_interactions structure
        """
        steps = completed_steps.get('steps', [])
        enabled_interactions = {'steps': []}

        for step in steps:
            step_id = step.get('id')
            adaptive_fields = step.get('adaptive_fields', {})

            # Check which media is present (non-empty string)
            has_teach_pendant = bool(adaptive_fields.get('image_teach_pendant', '').strip())
            has_cobot = bool(adaptive_fields.get('image_cobot', '').strip())
            has_video = bool(adaptive_fields.get('video', '').strip())

            # short_text and long_text are always present
            interaction_entry = {
                'step_id': step_id,
                'buttons': {
                    'short_text': True,
                    'long_text': True,
                    'teach_pendant': has_teach_pendant,
                    'cobot': has_cobot,
                    'video': has_video
                }
            }

            enabled_interactions['steps'].append(interaction_entry)

            logger.info(f"Step {step_id}: teach_pendant={has_teach_pendant}, "
                        f"cobot={has_cobot}, video={has_video}")

        logger.info(f"✓ Generated enabled_interactions for {len(steps)} steps")
        return enabled_interactions

    @staticmethod
    def save(data: Dict[str, Any], filepath: str) -> bool:
        """
        Save enabled_interactions to JSON file.

        Args:
            data: Enabled interactions dictionary
            filepath: Path where to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Enabled interactions saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save enabled_interactions: {e}")
            return False

    @staticmethod
    def load(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load enabled_interactions from JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Loaded dictionary if successful, None otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded enabled_interactions from {filepath}")
            return data
        except Exception as e:
            logger.error(f"✗ Failed to load enabled_interactions: {e}")
            return None


__all__ = ['EnabledInteractionsGenerator']