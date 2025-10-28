# utils/historical_data_manager.py
"""
Manages historical user interaction data for training the adaptive system.
Reads CSV logs and provides relevant context to the AI for each step.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """Manages historical interaction data from CSV logs."""

    def __init__(self, csv_path: str = 'interaction_logs.csv'):
        """
        Initialize the manager with a CSV file path.

        Args:
            csv_path: Path to the interaction logs CSV
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()

    def load_data(self) -> bool:
        """
        Load interaction data from CSV.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"✓ Loaded {len(self.df)} interaction records")
            return True
        except FileNotFoundError:
            logger.warning(f"CSV file not found: {self.csv_path}")
            self.df = pd.DataFrame()
            return False
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            self.df = pd.DataFrame()
            return False

    def get_user_history(self, user_id: str, current_step: int) -> List[Dict[str, Any]]:
        """
        Get interaction history for a specific user up to (but not including) current_step.

        Args:
            user_id: The experiment/user ID
            current_step: Current step number (excluded from history)

        Returns:
            List of dictionaries with step interactions
        """
        if self.df is None or self.df.empty:
            return []

        # Filter for this user and steps before current
        user_data = self.df[
            (self.df['experiment_id'] == user_id) &
            (self.df['step_id'] < current_step)
            ]

        # Group by step and aggregate what was viewed
        history = []
        for step_id in sorted(user_data['step_id'].unique()):
            step_data = user_data[user_data['step_id'] == step_id]

            # Get the last row for this step (final state)
            last_interaction = step_data.iloc[-1]

            step_summary = {
                'step_id': int(step_id),
                'step_name': last_interaction.get('step_name', 'Unknown'),
                'viewed': {
                    'short_text': bool(last_interaction.get('short_text_viewed', 0)),
                    'long_text': bool(last_interaction.get('long_text_viewed', 0)),
                    'single_pieces': bool(last_interaction.get('single_pieces_viewed', 0)),
                    'assembled_pieces': bool(last_interaction.get('assembly_viewed', 0)),
                    'video': bool(last_interaction.get('video_viewed', 0))
                }
            }
            history.append(step_summary)

        return history

    def get_all_users_history(self, exclude_user_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get interaction history for ALL users (useful for new user initialization).

        Args:
            exclude_user_id: Optional user ID to exclude (e.g., current user)

        Returns:
            Dictionary mapping user_id -> list of step interactions
        """
        if self.df is None or self.df.empty:
            return {}

        all_history = {}

        # Get unique user IDs
        user_ids = self.df['experiment_id'].unique()

        for user_id in user_ids:
            if exclude_user_id and user_id == exclude_user_id:
                continue

            # Get complete history for this user (all steps)
            user_data = self.df[self.df['experiment_id'] == user_id]

            user_history = []
            for step_id in sorted(user_data['step_id'].unique()):
                step_data = user_data[user_data['step_id'] == step_id]
                last_interaction = step_data.iloc[-1]

                step_summary = {
                    'step_id': int(step_id),
                    'step_name': last_interaction.get('step_name', 'Unknown'),
                    'viewed': {
                        'short_text': bool(last_interaction.get('short_text_viewed', 0)),
                        'long_text': bool(last_interaction.get('long_text_viewed', 0)),
                        'single_pieces': bool(last_interaction.get('single_pieces_viewed', 0)),
                        'assembled_pieces': bool(last_interaction.get('assembly_viewed', 0)),
                        'video': bool(last_interaction.get('video_viewed', 0))
                    }
                }
                user_history.append(step_summary)

            all_history[user_id] = user_history

        return all_history

    def get_aggregated_preferences_by_step(self, step_id: int) -> Dict[str, float]:
        """
        Get aggregated preferences across all users for a specific step.
        Returns percentage of users who viewed each content type.

        Args:
            step_id: The step ID to analyze

        Returns:
            Dictionary with content_type -> percentage (0.0 to 1.0)
        """
        if self.df is None or self.df.empty:
            return {}

        step_data = self.df[self.df['step_id'] == step_id]

        if step_data.empty:
            return {}

        total_users = len(step_data['experiment_id'].unique())

        aggregated = {
            'short_text': step_data['short_text_viewed'].sum() / total_users,
            'long_text': step_data['long_text_viewed'].sum() / total_users,
            'single_pieces': step_data['single_pieces_viewed'].sum() / total_users,
            'assembled_pieces': step_data['assembly_viewed'].sum() / total_users,
            'video': step_data['video_viewed'].sum() / total_users
        }

        return aggregated

    def format_for_prompt(self, user_id: str, current_step: int) -> str:
        """
        Format historical data for inclusion in AI prompt.

        Args:
            user_id: Current user ID
            current_step: Current step number

        Returns:
            Formatted string for prompt
        """
        # Get current user's history
        user_history = self.get_user_history(user_id, current_step)

        # Get all previous users' history (if this is step 1, or we want global context)
        all_users_history = self.get_all_users_history(exclude_user_id=user_id)

        prompt_text = ""

        # Current user history
        if user_history:
            prompt_text += f"CURRENT USER ({user_id}) PREVIOUS INTERACTIONS:\n"
            for step in user_history:
                viewed_items = [k for k, v in step['viewed'].items() if v]
                prompt_text += f"  Step {step['step_id']} ({step['step_name']}): viewed {', '.join(viewed_items)}\n"
            prompt_text += "\n"

        # Other users' aggregated patterns
        if all_users_history and current_step == 1:
            prompt_text += "PATTERNS FROM PREVIOUS USERS:\n"

            # Aggregate by step across all users
            step_patterns = {}
            for uid, history in all_users_history.items():
                for step in history:
                    sid = step['step_id']
                    if sid not in step_patterns:
                        step_patterns[sid] = {'short_text': [], 'long_text': [],
                                              'single pieces': [], 'assembled_pieces': [], 'video': []}

                    for content_type, viewed in step['viewed'].items():
                        step_patterns[sid][content_type].append(viewed)

            # Calculate percentages
            for sid in sorted(step_patterns.keys())[:5]:  # Show first 5 steps
                patterns = step_patterns[sid]
                prompt_text += f"  Step {sid}: "
                for content_type, views in patterns.items():
                    pct = sum(views) / len(views) * 100 if views else 0
                    if pct > 50:  # Only mention if majority viewed
                        prompt_text += f"{content_type}({pct:.0f}%) "
                prompt_text += "\n"

        return prompt_text


__all__ = ['HistoricalDataManager']