import datetime
import enum
import logging
from functools import cached_property

import pandas as pd
from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)


class Content(enum.Enum):
    SHORT_TEXT = 'short_text'
    LONG_TEXT = 'long_text'
    SINGLE_PIECES = 'single_pieces'
    ASSEMBLED_PIECES = 'assembled_pieces'
    VIDEO = 'video'


class Action(enum.Enum):
    START_EXPERIMENT = 'start_experiment'
    VIEW_SHORT_TEXT = 'toggle_short_text_block'
    VIEW_LONG_TEXT = 'toggle_long_text_block'
    VIEW_SINGLE_PIECES = 'toggle_single_pieces_block'
    VIEW_ASSEMBLED_PIECES = 'toggle_assembly_block'
    VIEW_VIDEO = 'toggle_video_block'
    NEXT_STEP = 'navigate_next'
    PREV_STEP = 'navigate_previous'
    STEP_LOADED = 'step_loaded'
    RESTART = 'restart_application'
    END = 'navigate_to_thankyou'


class LogEntry(BaseModel):
    experiment_id: str
    timestamp: datetime.datetime
    action: Action
    step_id: int
    step_name: str
    short_text_viewed: bool
    long_text_viewed: bool
    single_pieces_viewed: bool
    assembly_viewed: bool
    video_viewed: bool


class Step(BaseModel):
    displayed_content: list[Content]
    viewed_content: list[Content]


class UserHistory(BaseModel):
    experiment_id: str
    steps: dict[int, Step]

    @computed_field
    @cached_property
    def overall_viewed_content(self) -> dict[Content, int]:
        # Aggregate overall viewed content across all steps
        overall_viewed_content = {content: 0 for content in Content}

        for step in self.steps.values():
            for content in set(step.viewed_content) | set(step.displayed_content):
                overall_viewed_content[content] += 1
        return overall_viewed_content


class History(BaseModel):
    users: dict[str, UserHistory]


class HistoricalDataManager:

    def __init__(self, csv_path: str = 'interaction_logs.csv'):
        self.csv_path = csv_path
        self.history = self._build_history()

    def _load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path)
            df['experiment_id'] = df['experiment_id'].astype(str)
            df['step_id'] = df['step_id'].fillna(0).astype(int)
            df['short_text_viewed'] = df['short_text_viewed'].astype(bool)
            df['long_text_viewed'] = df['long_text_viewed'].astype(bool)
            df['single_pieces_viewed'] = df['single_pieces_viewed'].astype(bool)
            df['assembly_viewed'] = df['assembly_viewed'].astype(bool)
            df['video_viewed'] = df['video_viewed'].astype(bool)
            logger.info(f"✓ Loaded {len(df)} interaction records")
            return df
        except FileNotFoundError:
            logger.warning(f"CSV file not found: {self.csv_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def _build_history(self) -> History:
        df = self._load_data()
        if df is None or df.empty:
            return History(users={})

        all_histories = {}
        # Group by experiment_id
        for user_id, user_data in df.groupby('experiment_id'):
            steps = dict()
            for step_id, step_data in user_data.groupby('step_id'):
                if step_id == 0:
                    continue
                step_content = []
                displayed_content = []
                viewed_content = []
                for _, interaction_row in step_data.iterrows():
                    interaction = LogEntry(**interaction_row.to_dict())

                    match interaction.action:
                        case Action.STEP_LOADED:
                            if interaction.short_text_viewed:
                                displayed_content.append(Content.SHORT_TEXT)
                            if interaction.long_text_viewed:
                                displayed_content.append(Content.LONG_TEXT)
                            if interaction.single_pieces_viewed:
                                displayed_content.append(Content.SINGLE_PIECES)
                            if interaction.assembly_viewed:
                                displayed_content.append(Content.ASSEMBLED_PIECES)
                            if interaction.video_viewed:
                                displayed_content.append(Content.VIDEO)
                        case Action.VIEW_SHORT_TEXT:
                            viewed_content.append(Content.SHORT_TEXT)
                        case Action.VIEW_LONG_TEXT:
                            viewed_content.append(Content.LONG_TEXT)
                        case Action.VIEW_SINGLE_PIECES:
                            viewed_content.append(Content.SINGLE_PIECES)
                        case Action.VIEW_ASSEMBLED_PIECES:
                            viewed_content.append(Content.ASSEMBLED_PIECES)
                        case Action.VIEW_VIDEO:
                            viewed_content.append(Content.VIDEO)
                        case Action.PREV_STEP:
                            # in case of backward navigation, store the current state and reset
                            step_content.append((displayed_content, viewed_content))
                            displayed_content = []
                            viewed_content = []

                step_content.append((displayed_content, viewed_content))

                # In case of multiple interactions per step, store the last VALID state
                # (ignoring players who navigated back and forth without viewing anything)
                for displayed_content, viewed_content in step_content[::-1]:
                    if displayed_content or viewed_content:
                        steps[step_id] = Step(
                            displayed_content=displayed_content,
                            viewed_content=viewed_content
                        )
                        break

            all_histories[user_id] = UserHistory(
                experiment_id=user_id,
                steps=steps
            )

        self.history = History(users=all_histories)
        return self.history

    def get_user_history(self, user_id: str, current_step: int) -> UserHistory:
        """
        Get interaction history for a specific user up to (but not including) current_step.

        Args:
            user_id: The experiment/user ID
            current_step: Current step number (excluded from history)

        Returns:
            List of dictionaries with step interactions
        """
        user_history = self.history.users[user_id]
        return UserHistory(
            experiment_id=user_history.experiment_id,
            steps={k: v for k, v in user_history.steps.items() if k < current_step}
        )

    def get_aggregated_preferences_for_step(self, step_id: int) -> dict[Content, float]:
        """
        Get aggregated preferences across all users for a specific step.
        Returns percentage of users who viewed each content type.

        Args:
            step_id: The step ID to analyze

        Returns:
            Dictionary with content_type -> percentage (0.0 to 1.0)
        """

        overall_viewed_content = {content: 0 for content in Content}

        for user_id in self.history.users:
            user_history = self.history.users[user_id]
            if step_id in user_history.steps:
                step = user_history.steps[step_id]
                for content in set(step.viewed_content) | set(step.displayed_content):
                    overall_viewed_content[content] += 1

        total_users = len(self.history.users)

        return {content: (count / total_users) if total_users > 0 else 0.0
                for content, count in overall_viewed_content.items()}

    def get_aggregated_preferences_by_step(self, exclude_user_ids=None) -> dict[int, dict[Content, float]]:
        """
        Get aggregated preferences across all users for each step.

        :return: Dictionary with step_id -> (content_type -> percentage)
        """
        aggregated_preferences = {}

        for user_history in self.get_all_users_history(exclude_user_ids).users.values():
            for step_id, step in user_history.steps.items():
                if step_id not in aggregated_preferences:
                    aggregated_preferences[step_id] = {}
                aggregated_preferences[step_id] = self.get_aggregated_preferences_for_step(step_id)

        return aggregated_preferences

    def get_all_users_history(self, exclude_user_ids=None) -> History:
        """
        Get interaction history for ALL users (useful for new user initialization).

        Args:
            exclude_user_ids: Optional list of user IDs to exclude (e.g., current user)

        Returns:
            A History object with all users' histories
        """

        if exclude_user_ids is None:
            return self.history

        return History(users={
            user_id: self.history.users[user_id]
            for user_id in self.history.users if user_id not in exclude_user_ids
        })

    def format_for_prompt(self, user_id: str, current_step: int) -> str:
        """
        Format historical data for inclusion in AI prompt.

        Args:
            user_id: Current user ID
            current_step: Current step number

        Returns:
            Formatted string for prompt
        """

        prompt_text = ""

        # Get current user's history
        user_history = self.get_user_history(user_id, current_step)

        # Current user history
        if user_history:
            prompt_text += f"CURRENT USER ({user_id}) PREVIOUS INTERACTIONS:\n"
            for step_id, step_content in user_history.steps.items():
                viewed_items = [c.value for c in step_content.viewed_content]
                displayed_items = [c.value for c in step_content.displayed_content]
                prompt_text += f"  Step {step_id}: displayed [{', '.join(displayed_items)}] viewed [{', '.join(viewed_items)}]\n"
            prompt_text += "\n"

        # Other users' aggregated patterns
        preferences_by_step = self.get_aggregated_preferences_by_step(exclude_user_ids=[user_id])
        for step_id, preferences_for_step in preferences_by_step.items():
            prompt_text += f"  Step {step_id}: "
            for content_type, views in preferences_for_step.items():
                pct = views / sum([views for views in preferences_for_step.values()]) * 100 if views else 0
                if pct > 50:  # Only mention if majority viewed
                    prompt_text += f"{content_type.value}({pct:.0f}%) "
            prompt_text += "\n"

        return prompt_text


__all__ = ['HistoricalDataManager', 'History', 'UserHistory', 'Step', 'Content', 'Action']

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    manager = HistoricalDataManager('../first_round_of_test.csv')
    print('full history\n', manager.history, '\n')
    print('user 1 history\n', manager.history.users['1'], '\n')
    print('user 1 overall preferences\n', manager.history.users['1'].overall_viewed_content, '\n')
    print('user 1 history until step\n', manager.get_user_history('1', 4), '\n')
    print('aggregated preferences for step 2\n', manager.get_aggregated_preferences_for_step(2), '\n')
    print('aggregated preferences by step (all users)\n', manager.get_aggregated_preferences_by_step(), '\n')
    print('prompt format for user 1 at step 4\n', manager.format_for_prompt('1', 4), '\n')
