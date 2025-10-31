# utils/historical_data_manager.py

import datetime
import enum
import logging
from functools import cached_property
from typing import Optional, List

import pandas as pd
from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)


class Content(enum.Enum):
    SHORT_TEXT = 'short_text'
    LONG_TEXT = 'long_text'
    SINGLE_PIECES = 'single_pieces'
    ASSEMBLED_PIECES = 'assembly'  # ⚠️ Cambiato per match con CSV
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
    # Azioni aggiuntive che ignoriamo
    INITIAL_SUGGESTION = 'initial_suggestion'
    COMPUTED_SUGGESTION = 'computed_suggestion'
    SENTIENT_STEP_ADAPTED = 'sentient_step_adapted'


class Step(BaseModel):
    displayed_content: list[Content]
    viewed_content: list[Content]


class UserHistory(BaseModel):
    experiment_id: str
    steps: dict[int, Step]

    @computed_field
    @cached_property
    def overall_viewed_content(self) -> dict[Content, int]:
        overall_viewed_content = {content: 0 for content in Content}
        for step in self.steps.values():
            for content in set(step.viewed_content) | set(step.displayed_content):
                overall_viewed_content[content] += 1
        return overall_viewed_content


class History(BaseModel):
    users: dict[str, UserHistory]


class HistoricalDataManager:
    """
    Gestisce lo storico delle interazioni con reveal progressivo.

    Il CSV first_round_of_test.csv contiene i dati di 10 utenti.
    Quando un nuovo utente fa training, vengono rivelati solo:
    - Dati completi degli utenti precedenti (ID < current_user_id)
    - Dati parziali dell'utente corrente (step < current_step)
    """

    def __init__(self, historical_csv_path: str = 'first_round_of_test.csv'):
        """
        Args:
            historical_csv_path: Path al CSV con dati storici (formato: sep=';')
        """
        self.historical_csv_path = historical_csv_path
        self.full_historical_data = self._load_historical_data()

    def _load_historical_data(self) -> pd.DataFrame:
        """Carica il CSV storico completo (ma non lo espone tutto subito)"""
        try:
            # ⚠️ IMPORTANTE: Il CSV usa ';' come separatore
            df = pd.read_csv(self.historical_csv_path, sep=';')

            # Normalizza i nomi delle colonne
            df.columns = df.columns.str.strip().str.lower()

            # Converti i tipi
            df['experiment_id'] = df['experiment_id'].astype(str)
            df['step_id'] = df['step_id'].fillna(0).astype(int)

            # ⚠️ Le colonne nel CSV sono nominate diversamente
            view_columns = {
                'short_text_viewed': 'short_text_viewed',
                'long_text_viewed': 'long_text_viewed',
                'single_pieces_viewed': 'single_pieces_viewed',
                'assembly_viewed': 'assembly_viewed',
                'video_viewed': 'video_viewed'
            }

            for col in view_columns.values():
                if col in df.columns:
                    df[col] = df[col].astype(bool)
                else:
                    logger.warning(f"Column {col} not found in CSV")
                    df[col] = False

            logger.info(f"✓ Loaded {len(df)} historical records from {len(df['experiment_id'].unique())} users")
            return df

        except FileNotFoundError:
            logger.warning(f"Historical CSV not found: {self.historical_csv_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading historical CSV: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_available_history(self, current_user_id: str, current_step: int) -> History:
        """
        Ritorna SOLO lo storico disponibile fino a questo punto.

        Regole:
        - Utenti con ID < current_user_id: tutti gli step
        - Utente corrente: solo step < current_step
        - Utenti con ID > current_user_id: nessun dato (futuro)

        Args:
            current_user_id: ID utente corrente (es: "1", "2", "3")
            current_step: Step corrente (es: 5)

        Returns:
            History object con solo i dati "disponibili"
        """
        if self.full_historical_data.empty:
            return History(users={})

        try:
            current_user_num = int(current_user_id)
        except ValueError:
            # Se l'ID non è numerico, considera tutti gli utenti storici disponibili
            current_user_num = 999999

        # Filtra i dati disponibili
        available_data = self.full_historical_data[
            # Condizione 1: Utenti completati (ID < current)
            (self.full_historical_data['experiment_id'].astype(int) < current_user_num) |
            # Condizione 2: Utente corrente, solo step precedenti
            (
                    (self.full_historical_data['experiment_id'].astype(int) == current_user_num) &
                    (self.full_historical_data['step_id'] < current_step)
            )
            ].copy()

        logger.info(
            f"📊 Available history: {len(available_data)} records for user {current_user_id} at step {current_step}")

        # Costruisci History da questi dati
        return self._build_history_from_df(available_data)

    def _build_history_from_df(self, df: pd.DataFrame) -> History:
        """Costruisce oggetto History da DataFrame filtrato"""
        if df.empty:
            return History(users={})

        all_histories = {}

        for user_id, user_data in df.groupby('experiment_id'):
            steps = {}

            for step_id, step_rows in user_data.groupby('step_id'):
                if step_id == 0:
                    continue

                # Prendi l'ultima riga per questo step (in caso di navigate_previous)
                step_row = step_rows.iloc[-1]

                # Determina cosa è stato visualizzato
                viewed_content = []
                displayed_content = []  # Nel CSV non abbiamo displayed, solo viewed

                if step_row.get('short_text_viewed', False):
                    viewed_content.append(Content.SHORT_TEXT)
                if step_row.get('long_text_viewed', False):
                    viewed_content.append(Content.LONG_TEXT)
                if step_row.get('single_pieces_viewed', False):
                    viewed_content.append(Content.SINGLE_PIECES)
                if step_row.get('assembly_viewed', False):
                    viewed_content.append(Content.ASSEMBLED_PIECES)
                if step_row.get('video_viewed', False):
                    viewed_content.append(Content.VIDEO)

                steps[int(step_id)] = Step(
                    displayed_content=displayed_content,
                    viewed_content=viewed_content
                )

            all_histories[str(user_id)] = UserHistory(
                experiment_id=str(user_id),
                steps=steps
            )

        return History(users=all_histories)

    def get_user_history(self, user_id: str, current_step: int) -> UserHistory:
        """
        Storia dell'utente corrente fino allo step precedente.
        """
        available_history = self.get_available_history(user_id, current_step)

        if user_id not in available_history.users:
            return UserHistory(experiment_id=user_id, steps={})

        return available_history.users[user_id]

    def get_aggregated_preferences_for_step(self, user_id: str, current_step: int, target_step: int) -> dict[
        Content, float]:
        """
        Preferenze aggregate per uno step specifico, basate SOLO su utenti precedenti.

        Args:
            user_id: ID utente corrente
            current_step: Step corrente (per determinare cosa è disponibile)
            target_step: Step per cui calcolare le preferenze aggregate

        Returns:
            Dict con percentuali di visualizzazione per ogni tipo di contenuto
        """
        available_history = self.get_available_history(user_id, current_step)

        overall_viewed_content = {content: 0 for content in Content}
        total_users = 0

        for other_user_id, other_user_history in available_history.users.items():
            # Conta solo utenti precedenti per le aggregazioni
            if other_user_id == user_id:
                continue

            if target_step in other_user_history.steps:
                total_users += 1
                step = other_user_history.steps[target_step]
                for content in set(step.viewed_content) | set(step.displayed_content):
                    overall_viewed_content[content] += 1

        return {
            content: (count / total_users) if total_users > 0 else 0.0
            for content, count in overall_viewed_content.items()
        }

    def format_for_prompt(self, user_id: str, current_step: int) -> str:
        """
        Formatta lo storico disponibile per il prompt di Gemini.
        """
        available_history = self.get_available_history(user_id, current_step)

        prompt_text = ""

        # Storia utente corrente
        if user_id in available_history.users:
            user_history = available_history.users[user_id]
            if user_history.steps:
                prompt_text += f"CURRENT USER ({user_id}) PREVIOUS INTERACTIONS:\n"
                for step_id, step_content in sorted(user_history.steps.items()):
                    viewed_items = [c.value for c in step_content.viewed_content]
                    prompt_text += f"  Step {step_id}: viewed [{', '.join(viewed_items)}]\n"
                prompt_text += "\n"
            else:
                prompt_text += f"CURRENT USER ({user_id}): No previous interactions\n\n"
        else:
            prompt_text += f"CURRENT USER ({user_id}): No previous interactions\n\n"

        # Pattern aggregati altri utenti
        prompt_text += "OTHER USERS' AGGREGATED PATTERNS:\n"

        other_users_count = len([uid for uid in available_history.users.keys() if uid != user_id])

        if other_users_count > 0:
            prompt_text += f"  Based on data from {other_users_count} previous users:\n"

            # Calcola pattern per alcuni step chiave
            for target_step in range(1, min(current_step, 17)):  # Max 16 step
                prefs = self.get_aggregated_preferences_for_step(user_id, current_step, target_step)

                high_prefs = {c: p for c, p in prefs.items() if p > 0.3}
                if high_prefs:
                    items = [f"{c.value}({p * 100:.0f}%)" for c, p in high_prefs.items()]
                    prompt_text += f"    Step {target_step}: {', '.join(items)}\n"
        else:
            prompt_text += "  No historical data available from other users\n"

        return prompt_text


__all__ = ['HistoricalDataManager', 'History', 'UserHistory', 'Step', 'Content', 'Action']