import datetime
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from services.sentient_gemini_api import initial_style_recommendations, adapt_step
import pathlib
import dash
import dash_bootstrap_components as dbc
import flask
import pandas as pd
from dash import dcc, html, Input, Output, State
from utils.historical_data_manager import HistoricalDataManager
import shutil
from pathlib import Path

historical_manager = HistoricalDataManager('first_round_of_test.csv')
# Initialize the app with Flask server to handle static files


server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
                ],
                assets_folder='assets',
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0'}])


# Add route to serve dynamically generated CSS
@server.route('/assets/<path:path>')
def serve_assets(path):
    return flask.send_from_directory('./assets', path)


# Create routes for serving static files
@server.route('/images_single_pieces/<path:path>')
def serve_single_pieces(path):
    return flask.send_from_directory('./images_single_pieces', path)


@server.route('/images_assembly/<path:path>')
def serve_assembly(path):
    return flask.send_from_directory('./images_assembly', path)


@server.route('/images/assembly_process/<path:path>')
def serve_assembly_process(path):
    return flask.send_from_directory('./images/assembly_process', path)


@server.route('/videos/<path:path>')
def serve_videos(path):
    return flask.send_from_directory('./videos', path)


# Load JSON data
def load_assembly_process():
    with open('settings/steps_sources.json', 'r') as f:
        data = json.load(f)
    return data['assembly_process']


# Initialize log DataFrame
def init_log_df():
    if os.path.exists('interaction_logs.csv'):
        return pd.read_csv('interaction_logs.csv')
    else:
        return pd.DataFrame(columns=['experiment_id', 'timestamp', 'action', 'step_id', 'step_name'])


# Log user interaction
def log_interaction(experiment_id, mode, action, step_id=None, step_name=None, button_states=None):
    df = init_log_df()

    # not the best solution :)
    dropdown_options = [
        {'label': 'Data Collection', 'value': 'initial_visibility_data_collection.json'},
        {'label': 'Dynamically Adaptive', 'value': 'initial_visibility_dynamically_adaptive.json'},
        {'label': 'Rule-Based Adaptive', 'value': 'initial_visibility_rule_based_adaptive.json'},
        {'label': 'Static', 'value': 'initial_visibility_static_mode.json'},
        {'label': 'Sentient', 'value': 'sentient.json'}
    ]
    mode_to_label = {option['value']: option['label'] for option in dropdown_options}
    mode = mode_to_label.get(mode, 'Unknown Mode')

    # ✅ CRITICO: Inizializza SEMPRE tutti i campi viewed (anche se 0)
    new_row = {
        'experiment_id': experiment_id,
        'mode': mode,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        'action': action,
        'step_id': step_id if step_id is not None else 'N/A',
        'step_name': step_name if step_name is not None else 'N/A',
        # 🆕 SEMPRE presenti
        'short_text_viewed': 0,
        'long_text_viewed': 0,
        'single_pieces_viewed': 0,
        'assembly_viewed': 0,
        'video_viewed': 0
    }

    # ✅ Sovrascrivi con i valori effettivi se forniti
    if button_states:
        if isinstance(button_states, dict):
            # Caso 1: button_states è un dict con chiavi boolean
            new_row['short_text_viewed'] = 1 if button_states.get('short_text', False) else 0
            new_row['long_text_viewed'] = 1 if button_states.get('long_text', False) else 0
            new_row['single_pieces_viewed'] = 1 if button_states.get('single_pieces', False) else 0
            new_row['assembly_viewed'] = 1 if button_states.get('assembly', False) else 0
            new_row['video_viewed'] = 1 if button_states.get('video', False) else 0
        else:
            # Caso 2: button_states potrebbe essere passato in altro formato
            # (per retrocompatibilità con vecchie chiamate)
            for button_name, state in button_states.items():
                new_row[f'{button_name}_viewed'] = 1 if state else 0

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('interaction_logs.csv', index=False)

    return


# Inizializzazione del DataFrame di log con colonne aggiuntive
def init_log_df():
    columns = [
        'experiment_id', 'mode', 'timestamp', 'action', 'step_id', 'step_name',
        'short_text_viewed', 'long_text_viewed', 'single_pieces_viewed',
        'assembly_viewed', 'video_viewed'
    ]

    if os.path.exists('interaction_logs.csv'):
        df = pd.read_csv('interaction_logs.csv')
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        return df
    else:
        return pd.DataFrame(columns=columns)


# Funzione di utilità per ottenere lo stato completo dei pulsanti per il passo corrente
def get_complete_button_states(current_step, clicked_buttons):
    step_key = str(current_step)
    step_clicked = clicked_buttons.get(step_key, {})

    return {
        'short_text': step_clicked.get('short_text', False),
        'long_text': step_clicked.get('long_text', False),
        'single_pieces': step_clicked.get('single_pieces', False),
        'assembly': step_clicked.get('assembly', False),
        'video': step_clicked.get('video', False)
    }


def load_enabled_interactions():
    try:
        with open('settings/enabled_interactions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default to all buttons enabled if file not found
        return {"steps": [{"step_id": 1, "buttons": {
            "short_text": True,
            "long_text": True,
            "single_pieces": True,
            "assembly": True,
            "video": True
        }}]}


def load_initial_visibility():
    try:
        with open('settings/visibility/initial_visibility_data_collection.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default to all content hidden
        return {"steps": [{"step_id": 1, "content": {
            "short_text": False,
            "long_text": False,
            "single_pieces": False,
            "assembly": False,
            "video": False
        }}]}


def update_user_preferences(
        preferences: Dict[str, Dict[str, List[float]]],
        step_type: str,
        content_types: List[str],
        timestamp: Optional[float] = None,
        is_initial: bool = False
) -> Dict[str, Dict[str, List[float]]]:
    """
    Update user preferences with robust tracking per step type.

    Ensures that preferences are always tracked under the correct step type.
    """
    if preferences is None:
        preferences = {}

    # Ensure the step type exists in preferences
    if step_type not in preferences:
        preferences[step_type] = {}

    # Track each requested content type
    for content_type in content_types:
        if timestamp is None:
            timestamp = datetime.datetime.now().timestamp()

        if content_type not in preferences[step_type]:
            preferences[step_type][content_type] = []

        # For initial visibility, only add if no previous entries
        if not is_initial or not preferences[step_type][content_type]:
            preferences[step_type][content_type].append(timestamp)

    # Remove any 'Unknown' key if it exists
    if 'Unknown' in preferences:
        del preferences['Unknown']

    return preferences


def build_log_summary(preferences_store, step_type, clicked_buttons_for_step):
    # recent weighted preference of content types for the step category
    weighted = calculate_weighted_frequencies(preferences_store)  # already in your code
    recent_pref = (weighted.get(step_type) or {})
    return {
        "step_type": step_type,
        "recent_weighted": recent_pref,
        "clicked_now": clicked_buttons_for_step  # booleans for short/long/single/assembly/video
    }


def calculate_weighted_frequencies(
        preferences: Dict[str, Dict[str, List[float]]],
        decay_factor: float = 0.5,
        max_history: int = 20
) -> Dict[str, Dict[str, float]]:
    """
    Calculate weighted frequencies with decay for recent selections.

    Filters out any 'Unknown' type preferences.
    """
    if not preferences:
        return {}

    # Remove 'Unknown' type if present
    filtered_preferences = {
        k: v for k, v in preferences.items()
        if k != 'Unknown' and v
    }

    weighted_frequencies = {}

    for step_type, content_types in filtered_preferences.items():
        weighted_frequencies[step_type] = {}

        for content_type, timestamps in content_types.items():
            # Sort timestamps in descending order and take most recent
            recent_timestamps = sorted(timestamps, reverse=True)[:max_history]

            # Calculate weighted sum with exponential decay
            weighted_sum = sum(
                (decay_factor ** i) for i, _ in enumerate(recent_timestamps)
            )

            weighted_frequencies[step_type][content_type] = weighted_sum

    return weighted_frequencies


def as_markdown(s: str) -> str:
    if not s:
        return ""
    # force single newlines to render as <br>
    return s.replace("\n", "  \n")


def get_most_frequent_content(
        preferences: Dict[str, Dict[str, List[float]]],
        step_type: str
) -> Optional[str]:
    """
    Determine the most frequently requested content type for a specific step type.

    Args:
    - preferences: User preferences dictionary
    - step_type: Type of step to analyze

    Returns:
    Most frequent content type or None if no preferences exist
    """
    if not preferences or step_type not in preferences:
        return None

    # Calculate weighted frequencies
    weighted_freqs = calculate_weighted_frequencies(preferences)

    # Check if we have frequencies for this step type
    if step_type not in weighted_freqs or not weighted_freqs[step_type]:
        return None

    # Find and return the content type with the highest weighted frequency
    return max(
        weighted_freqs[step_type].items(),
        key=lambda x: x[1]
    )[0]


# Define placeholder image URL
placeholder_img = "https://developers.elementor.com/docs/assets/img/elementor-placeholder-image.png"

# Define custom styles to add to the layout
styles = {
    'full-view': {
        'width': '100%',
        'height': '100vh',
        'display': 'flex',
        'flexDirection': 'column',
        'overflow': 'hidden'
    },
    'intro-screen': {
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'background-color': '#f8f9fa',
        'flex': '1',
        'padding': '10px',
        'overflow-y': 'auto',
        'height': '100vh',
    },
    'training-screen': {
        'padding': '10px',
        'flex': '1',
        'display': 'flex',
        'flexDirection': 'column',
        'overflow-y': 'auto'
    },
    'content-container': {
        'flex': '1',
        'overflow-y': 'auto',
        'paddingBottom': '70px'
    },

    'text-content-area': {
        'min-height': '150px',
        'position': 'relative',
        'overflow': 'auto',
        'padding-bottom': '60px'
    },
    'image-container': {
        'position': 'relative',
        'height': '350px',
        'padding-bottom': '60px'
    },
    'image-wrapper': {
        'flex-grow': '1',
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'overflow': 'hidden',
        'position': 'relative',
        'height': 'calc(100% - 60px)'
    },
    'button-container': {
        'position': 'absolute',
        'bottom': '0',
        'left': '0',
        'width': '100%',
        'padding': '10px',
        'background-color': 'rgba(255, 255, 255, 0.9)',
        'z-index': '10'
    },
    'image-content': {
        'max-width': '100%',
        'max-height': '200px',
        'object-fit': 'contain'
    },
    'footer-container': {
        'position': 'fixed',
        'bottom': '0',
        'left': '0',
        'width': '100%',
        'padding': '10px',
        'backgroundColor': '#f8f9fa',
        'textAlign': 'center',
        'borderTop': '1px solid #e0e0e0',
        'z-index': '100'
    },
    'footer-content': {
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'flexDirection': 'column'
    },
    'footer-images': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'width': '100%',
        'marginTop': '5px'
    },
    'image-style': {
        'width': '80px',
        'height': 'auto'

    }}

# App layout
app.layout = html.Div([
    # Store components for state management
    dcc.Store(id='current-step', data=0),  # 0 = intro, 1+ = steps
    dcc.Store(id='experiment-id-store', data=None),
    dcc.Store(id='assembly-data-store', data=load_assembly_process()),
    dcc.Store(id='navigation-in-progress', data=False),  # Store to track navigation state
    dcc.Store(id='enabled-interactions-store', data=load_enabled_interactions()),
    dcc.Store(id='initial-visibility-store', data=load_initial_visibility()),
    dcc.Store(id='clicked-buttons-store', data={}),
    dcc.Store(id='user-preferences-store', data={}),
    dcc.Store(id='style-profile-token', data=None),
    dcc.Store(id='sentient-css-store', data=None),  # Store for CSS content

    # Introduction page
    html.Div(id='intro-container',
             style=styles['intro-screen'],
             children=[
                 html.Div(style={
                     'maxWidth': '900px',
                     'width': 'calc(100% - 40px)',
                     'padding': 'clamp(10px, 5%, 40px)',
                     'borderRadius': '15px',
                     'backgroundColor': 'white',
                     'boxShadow': '0 10px 25px rgba(0,0,0,0.1)',
                     'border': '1px solid #e1e4e8',
                     'margin': '200px auto 150px auto',
                     'boxSizing': 'border-box',
                     'position': 'relative',

                 }, children=[
                     html.H1("LEGO Assembly Training",
                             style={'text-align': 'center', 'margin-bottom': '30px', 'fontSize': '3rem',
                                    'fontWeight': '800', 'color': '#2c3e50', 'letterSpacing': '-1px'}),
                     html.P("Welcome! Enter your experiment ID to start your personalized training.",
                            style={'margin-bottom': '25px', 'fontSize': '1.3rem'}),
                     dbc.Input(id='experiment-id-input', type='text', placeholder='Enter Experiment ID',
                               style={'margin-bottom': '30px', 'fontSize': '1.2rem', 'height': '50px'}),

                     html.P("Choose how the interface guides you during training",
                            style={'margin-bottom': '10px', 'fontSize': '1.3rem', 'fontWeight': 'bold'}),
                     dcc.Dropdown(
                         id='visibility-mode-dropdown',
                         options=[
                             {'label': 'Data Collection – interaction data gathering mode',
                              'value': 'initial_visibility_data_collection.json'},
                             {'label': 'Static – standard interface with fixed instructions',
                              'value': 'initial_visibility_static_mode.json'},
                             {'label': 'Rule-Based Adaptive – rule-driven guidance based on historical data',
                              'value': 'initial_visibility_rule_based_adaptive.json'},
                             {'label': 'Dynamically Adaptive – real-time guidance based on current interactions',
                              'value': 'initial_visibility_dynamically_adaptive.json'},
                             {'label': 'Sentient – AI-powered personalized guidance based on your profile',
                              'value': 'sentient.json'}  #
                         ],
                         value='initial_visibility_data_collection.json',  # Default selection
                         clearable=False,
                         style={'text-align': 'center', 'margin-bottom': '20px', 'fontSize': '1.1rem',
                                'height': '50px'},
                         className="mb-4"
                     ),
                     html.Div(id='sentient-profile-form', style={'display': 'none'}, children=[
                         #     html.Hr(),
                         #     html.H5("Sentient Training Profile", className="text-primary mb-3"),

                         # Lingua
                         html.P([html.I(className="bi bi-translate me-2"), "Instruction Language"],
                                style={'fontSize': '1.3rem', 'fontWeight': 'bold'},
                                className="mb-2"),
                         dbc.Input(
                             id='profile-language',  # Manteniamo lo stesso ID per non rompere le callback
                             type='text',
                             placeholder='e.g., English, Italian, etc...',
                             style={
                                 'borderRadius': '10px',
                                 'borderColor': '#ced4da',
                                 'padding': '15px',
                                 'fontSize': '1.1rem',
                                 'height': '50px'
                             },
                             className="mb-4"
                         ),
                         html.Div([
                             html.P([html.I(className="bi bi-person-workspace me-2"),
                                     "Tell us about your history with LEGO: is this a familiar hobby for you or a brand-new experience?"],
                                    style={'fontSize': '1.3rem', 'fontWeight': 'bold', 'color': '#2c3e50'},
                                    className="mb-2"),
                             html.P(
                             #    "I am an expert, I love Technic.",
                                 style={'color': '#666', 'fontSize': '1.1rem'}),
                             dbc.Textarea(
                                 id='profile-experience',  # Maintaining this ID for general user narrative
                                 placeholder='e.g., I am an expert, I love Technic; Totally new, I am afraid to fail;...',
                                 style={'borderRadius': '12px', 'minHeight': '100px', 'fontSize': '1.1rem',
                                        'padding': '15px'}
                             ),
                             # Nuova domanda sul Setup Ambientale
                             html.Div([
                                 html.P([html.I(className="bi bi-display me-2"),
                                         "Where will your screen be positioned during assembly, and how easy will it be to read?"],
                                        style={'fontSize': '1.3rem', 'fontWeight': 'bold', 'color': '#2c3e50'},
                                        className="mb-2"),
                                 html.P(
                                     "Tell us if you'll hold it in your hand or if it will be on a table far from you.",
                                     style={'color': '#666', 'fontSize': '1.1rem'}),
                                 dbc.Textarea(
                                     id='distance-setup',  # ID univoco per questa domanda
                                     placeholder='e.g., On a table 1 meter away; In my hands; Propped up on a shelf...',
                                     style={'borderRadius': '12px', 'minHeight': '100px', 'fontSize': '1.1rem',
                                            'padding': '15px'}
                                 ),
                             ], className="mb-4"),
                         # dcc.Dropdown(
                         ], className="mb-4"),
                         #     id='profile-language',
                         #     options=[
                         #         {'label': 'Italiano', 'value': 'Italian'},
                         #         {'label': 'English', 'value': 'English'},
                         #         {'label': 'Deutsch', 'value': 'German'},
                         #         {'label': 'Français', 'value': 'French'}
                         #     ],
                         #     value='English',
                         #     style={'fontSize': '1.1rem', 'borderRadius': '8px'},
                         #     className="mb-3"
                         # ),

                         # Esperienza (Checklist Strategica)
                         # html.P([html.I(className="bi bi-person-workspace me-2"), "Prior Experience"],
                         #        "Prior Experience ", style={'fontSize': '1.3rem', 'fontWeight': 'bold'},
                         #        className="mb-2"),
                         # dbc.Checklist(
                         #     id='profile-experience-checklist',
                         #     style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '20px',
                         #            'border': '1px solid #dee2e6'},
                         #     options=[
                         #         {'label': html.Div([
                         #             html.B("Advanced LEGO Building:"),
                         #             html.Br(),
                         #             html.I(
                         #                 "I’ve assembled complex LEGO sets or Technic models with moving parts and gears.")
                         #         ]), 'value': 'lego_advanced'},
                         #         {'label': html.Div([
                         #             html.B("Industrial Assembly/Maintenance (Non-LEGO):"),
                         #             html.Br(),
                         #             html.I(
                         #                 "I’ve worked on mechanical assemblies or used tools while following technical instructions or diagrams.")
                         #         ]), 'value': 'industrial_mech'},
                         #         {'label': html.Div([
                         #             html.B("Warehouse Picking (Bin systems):"),
                         #             html.Br(),
                         #             html.I(
                         #                 "I’m familiar with locating items using bin or location codes (e.g., A1-B03).")
                         #         ]), 'value': 'warehouse_picking'},
                         #         {'label': html.Div([
                         #             html.B("No prior experience:"),
                         #             html.Br(),
                         #             html.I("I prefer clear, step-by-step guidance for every action.")
                         #         ]), 'value': 'none'}
                         #     ],
                         #     value=[],
                         #     label_style={'marginBottom': '15px', 'display': 'block', 'fontSize': '1.1rem'},
                         #     # Crea spazio tra le opzioni
                         #     input_style={'marginRight': '10px', 'transform': 'scale(1.2)'},
                         #     # Allontana e ingrandisce il quadratino
                         #     # id="profile-experience-checklist",
                         #
                         #     className="mb-3"
                         # ),
                         #
                         #
                         # # Obiettivo
                         # html.P("Training Objective", style={'fontSize': '1.3rem', 'fontWeight': 'bold'},
                         #        className="mb-2"),
                         # dcc.Dropdown(
                         #     id='profile-objective',
                         #     options=[
                         #         {'label': 'Focus on speed and efficiency', 'value': 'Speed'},
                         #         {'label': 'Focus on learning and precision', 'value': 'Learning'}
                         #     ],
                         #     value='Learning',
                         #     style={'fontSize': '1.1rem', 'borderRadius': '8px'},
                         #     className="mb-3"
                         # ),
                         #
                         # # Comfort Visivo
                         # html.P([html.I(className="bi bi-eye me-2"), "Visual Comfort & Accessibility"],
                         #        style={'fontSize': '1.3rem', 'fontWeight': 'bold'}, className="mb-2"),
                         # dbc.Checklist(
                         #     id='profile-visual-comfort',
                         #     options=[
                         #         {'label': 'High Contrast Mode', 'value': 'high_contrast'},
                         #         {'label': 'Large Text Mode', 'value': 'large_text'},
                         #         # {'label': 'Color-Blind Assist (Text labels for colors)', 'value': 'color_blind_assist'}
                         #     ],
                         #     value=[],
                         #     inline=True,
                         #     label_style={'fontSize': '1.1rem', 'marginRight': '20px'},
                         #     input_style={'transform': 'scale(1.2)', 'marginRight': '8px'},
                         #     className="mb-3"
                         # ),

                         # Note Libere
                         html.P("Any additional requests or comments for your personalized training?",
                                style={'fontSize': '1.3rem', 'fontWeight': 'bold'}, className="mb-2"),
                         dbc.Textarea(id='profile-other', placeholder='e.g. Prefer short sentences...',
                                      style={'borderRadius': '10px', 'borderColor': '#ced4da', 'padding': '15px'},
                                      className="mb-3"),
                     ]),
                     dcc.Store(id='sentient-profile-store', data=None),
                     dcc.Store(id='begin-button-loading', data=False),
                     dbc.Button(
                         [
                             dbc.Spinner(size="sm", spinner_class_name="me-2", id="begin-spinner",
                                         spinner_style={"display": "none"}),
                             html.Span("Begin Training", id="begin-button-text")
                         ],
                         id='begin-button',
                         color='primary',
                         style={'width': '100%', 'fontSize': '1.8rem', 'padding': '15px', 'fontWeight': 'bold',
                                'borderRadius': '12px', 'backgroundColor': '#007bff',
                                'boxShadow': '0 4px 15px rgba(0, 123, 255, 0.3)', 'transition': 'all 0.3s ease'}
                     )
                 ])
             ]),

    # Training steps page
    html.Div(id='training-container', style={'display': 'none'}, children=[
        html.Div(className="container-fluid", children=[
            # Header

            # Navigation buttons
            html.Div(className="d-flex justify-content-between align-items-center mb-1", children=[
                html.H3(id='step-header', className="m-0"),
                html.Div(className="d-flex gap-2", children=[
                    dbc.Button("PREVIOUS", id='prev-button', color='secondary',
                               style={'padding': '20px 20px', 'width': '150px'}),
                    dbc.Button(
                        [
                            dbc.Spinner(size="sm", spinner_class_name="me-2", id="next-spinner",
                                        spinner_style={"display": "none"}),
                            html.Span("NEXT", id="next-button-text")
                        ],
                        id='next-button',
                        color='primary',
                        style={'padding': '20px 20px', 'width': '150px'}),
                ])
            ]),

            # Top row: Text descriptions
            html.Div(className="row mb-1", children=[
                # Short text area
                html.Div(className="col-md-4", style={'height': '150px'}, children=[
                    html.Span("Short Description", className="h5 d-block mb-1"),
                    html.Div(style=styles['text-content-area'], children=[
                        html.Div(id="short-text-placeholder", className="placeholder-glow", children=[
                            html.Span(className="placeholder col-5"),
                            html.Span(className="placeholder col-3"),
                            html.Span(className="placeholder col-4"),
                            html.Span(className="placeholder col-4")
                        ]),
                        dcc.Markdown(id="short-text-content",
                                     style={'display': 'none'},
                                     link_target="_blank",
                                     dangerously_allow_html=True),
                        html.Div(style=styles['button-container'], children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="short-text-btn", color="primary", size="lg", style={"width": "100%"})
                        ])
                    ])
                ]),

                # Long text area
                html.Div(className="col-md-8", children=[
                    html.Span("Long Description", className="h5 d-block mb-1"),
                    html.Div(style=styles['text-content-area'], children=[
                        html.Div(id="long-text-placeholder", className="placeholder-glow", children=[
                            html.Span(className="placeholder col-7"),
                            html.Span(className="placeholder col-4"),
                            html.Span(className="placeholder col-6")
                        ]),
                        dcc.Markdown(id="long-text-content",
                                     style={'display': 'none'},
                                     link_target="_blank",
                                     dangerously_allow_html=True),
                        html.Div(style=styles['button-container'], children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="long-text-btn", color="primary", size="lg", style={"width": "100%"})
                        ])
                    ])
                ])
            ]),

            # Bottom row: Visual content
            html.Div(className="row", children=[
                # Individual Parts image
                html.Div(className="col-md-4 mb-4", children=[
                    html.Span("Single Components Image", className="h5 d-block mb-2"),
                    html.Div(style=styles['image-container'], children=[
                        html.Div(style=styles['image-wrapper'], children=[
                            html.Img(id="single-pieces-placeholder",
                                     src=placeholder_img,
                                     className="img-fluid",
                                     style=styles['image-content']),
                            html.Img(id="single-pieces-img",
                                     style={'display': 'none', **styles['image-content']},
                                     className="img-fluid")
                        ]),
                        html.Div(style=styles['button-container'], children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="single-pieces-btn", color="primary", size="lg", style={"width": "100%"})
                        ])
                    ])
                ]),

                # Assembled Parts image
                html.Div(className="col-md-4 mb-4", children=[
                    html.Span("Assembly Image", className="h5 d-block mb-2"),
                    html.Div(style=styles['image-container'], children=[
                        html.Div(style=styles['image-wrapper'], children=[
                            html.Img(id="assembly-placeholder",
                                     src=placeholder_img,
                                     className="img-fluid",
                                     style=styles['image-content']),
                            html.Img(id="assembly-img",
                                     style={'display': 'none', **styles['image-content']},
                                     className="img-fluid")
                        ]),
                        html.Div(style=styles['button-container'], children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="assembly-btn", color="primary", size="lg", style={"width": "100%"})
                        ])
                    ])
                ]),

                # Video section
                html.Div(className="col-md-4 mb-4", children=[
                    html.Span("Video", className="h5 d-block mb-2"),
                    html.Div(style=styles['image-container'], children=[
                        html.Div(style=styles['image-wrapper'], children=[
                            html.Img(id="video-placeholder",
                                     src=placeholder_img,
                                     className="img-fluid",
                                     style=styles['image-content']),
                            html.Video(id="video-player",
                                       controls=True,
                                       autoPlay=False,
                                       style={'display': 'none', **styles['image-content']},
                                       className="img-fluid")
                        ]),
                        html.Div(style=styles['button-container'], children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="video-btn", color="primary", size="lg", style={"width": "100%"})
                        ])
                    ])
                ])
            ]),

        ]),

    ]),
    # Thank you page
    html.Div(id='thankyou-container', style={'display': 'none'}, children=[
        html.Div(className="container-fluid d-flex flex-column justify-content-center align-items-center",
                 style={'height': '80vh'}, children=[
                html.H1("Thanks for Partecipating!",
                        className="mb-5 text-center"),
                html.Div(className="text-center", children=[
                    dbc.Button("Restart", id='restart-button', color='primary',
                               style={'padding': '20px 20px', 'width': '200px'})
                ])
            ])
    ]),

    html.Div(
        style=styles['footer-container'],
        children=[
            # Stack vertically
            html.Div(
                className="d-flex flex-column justify-content-start align-items-start",
                children=[
                    html.H5("Style explanation:", className="mb-1 text-start"),
                    html.H6(id='style-explanation', className='small text-muted mb-2'),

                    html.H5("Content explanation:", className="mb-1 text-start"),
                    html.H6(id='sentient-last-explanation', className='small text-muted mb-6'),
                ]
            ),

            html.Div(className="d-flex justify-content-center align-items-center", children=[
                # Experiment ID
                html.Div(id='experiment-id-display', className="me-3"),  # Add margin to separate

                # Step Counter with Progress Bar
                html.Div(id='step-counter', children=[
                    html.Div(className="d-flex align-items-center", children=[
                        dbc.Progress(id="step-progress-bar", value=0, style={"width": "200px", "height": "10px"}),
                        html.Span(id="step-text", className="ms-2 text-muted small")
                    ])
                ]),
            ]),

            # Images Section

            html.Div(style=styles['footer-images'], children=[
                # html.Img(src='/assets/logosps.png', style=styles['image-style']),  # Left Image
                # html.Img(src='/assets/logoxr.png', style=styles['image-style'])  # Right Image
            ]),
        ])
])

# Clientside callback to show spinner immediately on button click
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            return [{"display": "inline-block"}, true, "Loading..."];
        }
        return [{"display": "none"}, false, "Begin Training"];
    }
    """,
    [Output('begin-spinner', 'spinner_style'),
     Output('begin-button', 'disabled'),
     Output('begin-button-text', 'children')],
    [Input('begin-button', 'n_clicks')],
    prevent_initial_call=True
)

# Clientside callback to show spinner on NEXT button click
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            return [{"display": "inline-block"}, true, "Loading..."];
        }
        return [{"display": "none"}, false, "NEXT"];
    }
    """,
    [Output('next-spinner', 'spinner_style'),
     Output('next-button', 'disabled'),
     Output('next-button-text', 'children')],
    [Input('next-button', 'n_clicks')],
    prevent_initial_call=True
)

# Clientside callback to inject CSS into document head
app.clientside_callback(
    """
    function(css) {
        if (css) {
            // Remove existing sentient style if present
            var existingStyle = document.getElementById('sentient-dynamic-style');
            if (existingStyle) {
                existingStyle.remove();
            }
            // Create and inject new style element
            var styleEl = document.createElement('style');
            styleEl.id = 'sentient-dynamic-style';
            styleEl.textContent = css;
            document.head.appendChild(styleEl);
            console.log('Sentient CSS injected successfully');
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('sentient-css-store', 'data', allow_duplicate=True),
    [Input('sentient-css-store', 'data')],
    prevent_initial_call=True
)


# Callbacks
@app.callback(
    [Output('intro-container', 'style'),
     Output('training-container', 'style'),
     Output('experiment-id-store', 'data'),
     Output('current-step', 'data'),
     Output('sentient-profile-store', 'data'),
     Output('style-explanation', 'children'),
     Output('style-profile-token', 'data'),
     Output('sentient-css-store', 'data'),
     Output('begin-spinner', 'spinner_style', allow_duplicate=True),
     Output('begin-button', 'disabled', allow_duplicate=True),
     Output('begin-button-text', 'children', allow_duplicate=True)],
    [Input('begin-button', 'n_clicks')],
    [State('experiment-id-input', 'value'),
     State('visibility-mode-dropdown', 'value'),
     State('profile-language', 'value'),
     State('profile-experience', 'value'),  # Legge l'esperienza aperta
     State('distance-setup', 'value'),
     State('profile-other', 'value'),  # Legge le note/richieste libere
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def begin_training(n_clicks, experiment_id, mode, lang, experience, setup, other, assembly_data):
    # Normalise experiment id
    if not experiment_id:
        experiment_id = 'unknown'

    # Defaults
    profile = None
    style_expl = ""
    style_token = None
    css_text = ""
    css_html = ""

    # Sentient mode: build profile and request style overrides
    if mode == 'sentient.json':
        profile = {
            'language': lang,
            'prior_experience': experience if experience else ['None provided'],
            'screen_setup': setup if setup else 'Standard placement',  # AGGIUNTO
             'other_requests': (other or '').strip()
        }

        # Categories present in the current session
        categories = sorted(list(set(step.get('category', 'Unknown') for step in (assembly_data or []))))

        try:
            print(f"Calling initial_style_recommendations with profile={profile}, categories={categories}")
            out = initial_style_recommendations(profile, categories)

            css = out.get('css_overrides', "") or ""
            style_expl = out.get('explanation', "") or ""
            style_token = out.get('style_profile_token', "") or ""

            print(f"Received style_token: {style_token}")
            if css:
                # Persist to assets (for future page loads) and store raw CSS for immediate injection
                assets_dir = pathlib.Path('assets')
                assets_dir.mkdir(exist_ok=True)
                (assets_dir / 'sentient_overrides.css').write_text(css, encoding='utf-8')
                css_html = css  # Store raw CSS, clientside callback will inject it

                print("Wrote CSS to assets/sentient_overrides.css and prepared for injection.")
        except Exception as e:
            style_expl = f"Style recommendation failed; using defaults. Error: {str(e)}"
            import traceback
            print("Error in initial_style_recommendations:", traceback.format_exc())

    # Log start
    log_interaction(experiment_id, mode, 'start_experiment')

    # Switch to training view; step 1
    return (
        {'display': 'none'},
        {'display': 'block', **styles['training-screen']},
        experiment_id,
        1,
        profile,
        style_expl,
        style_token,
        css_html,
        {"display": "none"},  # Hide spinner after loading
        False,  # Re-enable button
        "Begin Training"  # Reset button text
    )


# Set navigation in progress
@app.callback(
    Output('navigation-in-progress', 'data'),
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')]
)
def set_navigation_in_progress(prev_clicks, next_clicks):
    if prev_clicks is None and next_clicks is None:
        return False
    return True


# Add a new callback to handle visibility mode selection
@app.callback(
    Output('initial-visibility-store', 'data'),
    [Input('visibility-mode-dropdown', 'value')]
)
def update_visibility_mode(selected_mode):
    try:
        with open(f'settings/visibility/{selected_mode}', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to data collection mode if file not found
        with open('settings/visibility/initial_visibility_data_collection.json', 'r') as f:
            return json.load(f)


def convert_aggregated_preferences(aggregated_prefs):
    """
    Converte le chiavi Content enum in stringhe per la compatibilità con Gemini

    Args:
        aggregated_prefs: Dict con chiavi Content enum

    Returns:
        Dict con chiavi stringa
    """
    if not aggregated_prefs:
        return {}

    converted = {}
    for content_type, percentage in aggregated_prefs.items():
        # Converti Content enum in stringa
        if hasattr(content_type, 'value'):
            key = content_type.value
        else:
            key = str(content_type)

        # Assicurati che percentage sia un numero
        if isinstance(percentage, (list, tuple)):
            # Se è una lista/tupla, prendi il primo elemento
            value = float(percentage[0]) if percentage else 0.0
        elif isinstance(percentage, (int, float)):
            value = float(percentage)
        else:
            value = 0.0

        converted[key] = value

    return converted


# ============================================================================
# CALLBACK update_step_content - VERSIONE CORRETTA
# ============================================================================

@app.callback(
    [Output('step-header', 'children'),
     Output('step-progress-bar', 'value'),
     Output('step-text', 'children'),
     Output('experiment-id-display', 'children'),
     Output('single-pieces-img', 'src'),
     Output('assembly-img', 'src'),
     Output('video-player', 'src'),
     Output('short-text-content', 'children'),
     Output('long-text-content', 'children'),
     Output('short-text-placeholder', 'style'),
     Output('short-text-content', 'style'),
     Output('long-text-placeholder', 'style'),
     Output('long-text-content', 'style'),
     Output('single-pieces-placeholder', 'style'),
     Output('single-pieces-img', 'style'),
     Output('assembly-placeholder', 'style'),
     Output('assembly-img', 'style'),
     Output('video-placeholder', 'style'),
     Output('video-player', 'style'),
     Output('sentient-last-explanation', 'children'),
     Output('next-spinner', 'spinner_style', allow_duplicate=True),
     Output('next-button', 'disabled', allow_duplicate=True),
     Output('next-button-text', 'children', allow_duplicate=True)],
    [Input('current-step', 'data')],
    [State('assembly-data-store', 'data'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('sentient-profile-store', 'data'),
     State('style-profile-token', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data'),
     State('navigation-in-progress', 'data')],
    prevent_initial_call=True
)
def update_step_content(current_step, assembly_data, experiment_id, mode, profile, style_token, clicked, prefs,
                        navigation_in_progress):
    # 🛑 CONTROLLO CRITICO: Impedisci l'esecuzione se la navigazione è in corso.
    # Questo filtro blocca le chiamate in cascata veloci che avvengono durante la navigazione.
    if navigation_in_progress is True:
        # PreventUpdate interrompe immediatamente il callback senza sprecare API
        raise dash.exceptions.PreventUpdate
    if current_step <= 0 or current_step > len(assembly_data):
        return "", 0, "", "", "", "", "", "", "", {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "", {
            "display": "none"}, False, "NEXT"
    enabled_interactions = load_enabled_interactions()
    step = assembly_data[current_step - 1]
    step_type = step.get('category', 'Unknown')

    # defaults from file
    title = step['name']
    af = step['adaptive_fields']
    sp, ap, vp = af['image_single_pieces'], af['image_assembly'], af['video']

    short_text, long_text = af['short_text'], af['long_text']

    # default visibility
    vis = {"short_text": False, "long_text": False, "single_pieces": False, "assembly": False, "video": False}
    explanation = ""

    # SENTIENT MODE: Adapt content per step
    if mode == 'sentient.json' and profile and style_token:
        try:
            # Genera il contesto storico formattato
            user_history_formatted = historical_manager.format_for_prompt(experiment_id, current_step)

            print(f"\n{'=' * 80}")
            print(f"USER {experiment_id} - STEP {current_step}")
            print(f"{'=' * 80}")
            print(user_history_formatted)
            print(f"{'=' * 80}\n")

            # Passa user_id e current_step per ottenere preferenze aggregate
            aggregated_prefs_raw = historical_manager.get_aggregated_preferences_for_step(
                experiment_id,
                current_step,
                current_step  # Preferenze per lo step corrente
            )
            aggregated_prefs = convert_aggregated_preferences(aggregated_prefs_raw)

            # Prepara il payload dello step
            step_payload = {
                "step_id": current_step,
                "name": title,
                "category": step_type,
                "adaptive_fields": af
            }

            print(f"Aggregated prefs for step {current_step}: {aggregated_prefs}")

            # Chiama adapt_step
            out = adapt_step(
                user_profile=profile,
                style_profile_token=style_token,
                step_payload=step_payload,
                user_history_formatted=user_history_formatted,
                aggregated_preferences=aggregated_prefs,
                enabled_interactions=enabled_interactions
            )

            print(f"Gemini adaptation: {out.get('explanation_of_changes', 'No explanation')}")

            # apply returned changes
            title = out.get('title', title)
            patch = out.get('adaptive_fields', {})
            short_text = patch.get('short_text', short_text)
            long_text = patch.get('long_text', long_text)
            sp = patch.get('image_single_pieces', sp)
            ap = patch.get('image_assembly', ap)
            vp = patch.get('video', vp)
            # ⚠️ CRITICAL: Valida e forza i vincoli DOPO la risposta di Gemini
            vis_suggested = out.get('initial_visibility', vis)
            vis = validate_and_enforce_visibility(
                vis_suggested,
                current_step,
                enabled_interactions,  # Ricarica la config
                step_type
            )

            print(f"After validation: {vis}")
            print(f"{'=' * 80}\n")

            explanation = out.get('explanation_of_changes', "")

            # log the adaptation
            log_interaction(experiment_id, mode, 'sentient_step_adapted', current_step, title, vis)

        except Exception as e:
            explanation = f"Adaptive update failed; using base content. Error: {str(e)}"
            import traceback
            print(f"\n!!! Error in adapt_step !!!")
            print(traceback.format_exc())

    progress_value = (current_step / len(assembly_data)) * 100
    step_text = f"Step {current_step} of {len(assembly_data)}"
    experiment_id_display = f"Experiment ID: {experiment_id}"

    short_text = as_markdown(short_text)
    long_text = as_markdown(long_text)

    def show(h):
        return {'display': 'block', **styles['image-content']} if h else {'display': 'none', **styles['image-content']}

    def show_txt(h):
        return {'display': 'block'} if h else {'display': 'none'}

    def hide_txt(h):
        return {'display': 'none'} if h else {'display': 'block'}

    return (
        f"{title}",
        progress_value,
        step_text,
        experiment_id_display,
        sp, ap, vp,
        short_text, long_text,
        hide_txt(vis['short_text']), show_txt(vis['short_text']),
        hide_txt(vis['long_text']), show_txt(vis['long_text']),
        show(not vis['single_pieces']), show(vis['single_pieces']),
        show(not vis['assembly']), show(vis['assembly']),
        show(not vis['video']), show(vis['video']),
        explanation,
        {"display": "none"},  # Hide spinner after loading
        False,  # Re-enable button
        "NEXT"  # Reset button text
    )


#  toggle_short_text callback

@app.callback(
    [Output('short-text-placeholder', 'style', allow_duplicate=True),
     Output('short-text-content', 'style', allow_duplicate=True),
     Output('short-text-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True),
     Output('user-preferences-store', 'data', allow_duplicate=True)],  # Added for adaptive mode
    [Input('short-text-btn', 'n_clicks')],
    [State('short-text-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_short_text(n_clicks, placeholder_style, experiment_id, mode, current_step,
                      assembly_data, clicked_buttons, user_preferences):
    if n_clicks is None:
        return placeholder_style, {'display': 'none'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
    step_type = assembly_data[current_step - 1].get('category', 'Unknown') if 0 < current_step <= len(
        assembly_data) else 'Unknown'

    is_showing = placeholder_style.get('display') == 'none'

    # Use string keys for steps
    step_key = str(current_step)
    if step_key not in clicked_buttons:
        clicked_buttons[step_key] = {}

    if is_showing:
        # Hide content
        clicked_buttons[step_key]['short_text'] = False
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_short_text_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['short_text'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_short_text_block", current_step, step_name, button_states)

        # Update user preferences with timestamp
        user_preferences = update_user_preferences(
            user_preferences, step_type, ['short_text'], datetime.now().timestamp(), is_initial=False)

        return {'display': 'none'}, {'display': 'block'}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons, user_preferences


# toggle long_text
@app.callback(
    [Output('long-text-placeholder', 'style', allow_duplicate=True),
     Output('long-text-content', 'style', allow_duplicate=True),
     Output('long-text-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True),
     Output('user-preferences-store', 'data', allow_duplicate=True)],  # Added for adaptive mode
    [Input('long-text-btn', 'n_clicks')],
    [State('long-text-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_long_text(n_clicks, placeholder_style, experiment_id, mode, current_step,
                     assembly_data, clicked_buttons, user_preferences):
    if n_clicks is None:
        return placeholder_style, {'display': 'none'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
    step_type = assembly_data[current_step - 1].get('category', 'Unknown') if 0 < current_step <= len(
        assembly_data) else 'Unknown'

    is_showing = placeholder_style.get('display') == 'none'

    # Update clicked buttons store
    step_key = str(current_step)
    if step_key not in clicked_buttons:
        clicked_buttons[step_key] = {}

    if is_showing:
        # Hide content
        clicked_buttons[step_key]['long_text'] = False
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_long_text_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['long_text'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_long_text_block", current_step, step_name, button_states)

        # Update user preferences with timestamp
        user_preferences = update_user_preferences(
            user_preferences, step_type, ['long_text'], datetime.now().timestamp(), is_initial=False)

        return {'display': 'none'}, {'display': 'block'}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons, user_preferences


# toggle single_pieces callback
@app.callback(
    [Output('single-pieces-placeholder', 'style', allow_duplicate=True),
     Output('single-pieces-img', 'style', allow_duplicate=True),
     Output('single-pieces-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True),
     Output('user-preferences-store', 'data', allow_duplicate=True)],  # Added for adaptive mode
    [Input('single-pieces-btn', 'n_clicks')],
    [State('single-pieces-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_single_pieces(n_clicks, placeholder_style, experiment_id, mode, current_step,
                         assembly_data, clicked_buttons, user_preferences):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
    step_type = assembly_data[current_step - 1].get('category', 'Unknown') if 0 < current_step <= len(
        assembly_data) else 'Unknown'

    is_showing = placeholder_style.get('display') == 'none'

    # Update clicked buttons store
    step_key = str(current_step)
    if step_key not in clicked_buttons:
        clicked_buttons[step_key] = {}

    if is_showing:
        # Hide content
        clicked_buttons[step_key]['single_pieces'] = False
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_single_pieces_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['single_pieces'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_single_pieces_block", current_step, step_name, button_states)

        # Update user preferences with timestamp
        user_preferences = update_user_preferences(
            user_preferences, step_type, ['single_pieces'], datetime.now().timestamp(), is_initial=False)

        return {'display': 'none'}, {'display': 'block', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons, user_preferences


# toggle assembly callback
@app.callback(
    [Output('assembly-placeholder', 'style', allow_duplicate=True),
     Output('assembly-img', 'style', allow_duplicate=True),
     Output('assembly-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True),
     Output('user-preferences-store', 'data', allow_duplicate=True)],  # Added for adaptive mode
    [Input('assembly-btn', 'n_clicks')],
    [State('assembly-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_assembly(n_clicks, placeholder_style, experiment_id, mode, current_step,
                    assembly_data, clicked_buttons, user_preferences):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
    step_type = assembly_data[current_step - 1].get('category', 'Unknown') if 0 < current_step <= len(
        assembly_data) else 'Unknown'

    is_showing = placeholder_style.get('display') == 'none'

    # Update clicked buttons store
    step_key = str(current_step)
    if step_key not in clicked_buttons:
        clicked_buttons[step_key] = {}

    if is_showing:
        # Hide content
        clicked_buttons[step_key]['assembly'] = False
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_assembly_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['assembly'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_assembly_block", current_step, step_name, button_states)

        # Update user preferences with timestamp
        user_preferences = update_user_preferences(
            user_preferences, step_type, ['assembly'], datetime.now().timestamp(), is_initial=False)

        return {'display': 'none'}, {'display': 'block', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons, user_preferences


# toggle video callback
@app.callback(
    [Output('video-placeholder', 'style', allow_duplicate=True),
     Output('video-player', 'style', allow_duplicate=True),
     Output('video-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True),
     Output('user-preferences-store', 'data', allow_duplicate=True),  # Added for adaptive mode
     Output('video-player', 'autoPlay')],
    [Input('video-btn', 'n_clicks')],
    [State('video-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_video(n_clicks, placeholder_style, experiment_id, mode, current_step,
                 assembly_data, clicked_buttons, user_preferences):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences, False

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
    step_type = assembly_data[current_step - 1].get('category', 'Unknown') if 0 < current_step <= len(
        assembly_data) else 'Unknown'

    is_showing = placeholder_style.get('display') == 'none'

    # Update clicked buttons store
    step_key = str(current_step)
    if step_key not in clicked_buttons:
        clicked_buttons[step_key] = {}

    if is_showing:
        # Hide content
        clicked_buttons[step_key]['video'] = False
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_video_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences, False
    else:
        # Show content, update preferences, and autoplay
        clicked_buttons[step_key]['video'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, mode, "toggle_video_block", current_step, step_name, button_states)

        # Update user preferences with timestamp
        user_preferences = update_user_preferences(
            user_preferences, step_type, ['video'], datetime.now().timestamp(), is_initial=False)

        return {'display': 'none'}, {'display': 'block', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons, user_preferences, True


# navigation callback
@app.callback(
    [Output('current-step', 'data', allow_duplicate=True),
     Output('navigation-in-progress', 'data', allow_duplicate=True),
     Output('training-container', 'style', allow_duplicate=True),
     Output('thankyou-container', 'style', allow_duplicate=True)],
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('initial-visibility-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('clicked-buttons-store', 'data'),
     State('training-container', 'style'),
     State('thankyou-container', 'style')],
    prevent_initial_call=True
)
def navigate_steps(prev_clicks, next_clicks, initial_visibility, current_step, assembly_data, experiment_id, mode,
                   clicked_buttons, training_style, thankyou_style):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_step, False, training_style, thankyou_style

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    # Get the current state of all buttons for logging
    button_states = get_complete_button_states(current_step, clicked_buttons)

    # If user clicks previous from the first step, do nothing
    if button_id == 'prev-button' and current_step > 1:
        log_interaction(experiment_id, mode, 'navigate_previous', current_step, step_name, button_states)
        # If we're returning from thank you page to the last step
        if current_step == len(assembly_data) + 1:
            return len(assembly_data), False, {'display': 'block', **styles['training-screen']}, {'display': 'none'}
        return current_step - 1, False, training_style, thankyou_style

    # If user clicks next on the last step, show thank you page
    elif button_id == 'next-button' and current_step == len(assembly_data):
        log_interaction(experiment_id, mode, 'navigate_to_thankyou', current_step, step_name, button_states)
        return len(assembly_data) + 1, False, {'display': 'none'}, {'display': 'block'}

    # Regular next button navigation
    elif button_id == 'next-button' and current_step < len(assembly_data):
        log_interaction(experiment_id, mode, 'navigate_next', current_step, step_name, button_states)

        # Defensive programming: ensure data exists
        if not initial_visibility or 'steps' not in initial_visibility:
            # Fallback to default visibility
            initial_visibility = {
                'steps': [{
                    'step_id': current_step,
                    'content': {
                        'short_text': False,
                        'long_text': False,
                        'single_pieces': False,
                        'assembly': False,
                        'video': False
                    }
                }]
            }

        # Find the configuration for the current step
        try:
            step_config = next(
                (step for step in initial_visibility['steps'] if step['step_id'] == current_step),
                initial_visibility['steps'][0]  # Default to first step if no match
            )
        except (IndexError, KeyError):
            # Fallback to default configuration
            step_config = {
                'content': {
                    'short_text': False,
                    'long_text': False,
                    'single_pieces': False,
                    'assembly': False,
                    'video': False
                }
            }

        step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
        log_interaction(experiment_id, mode, 'initial_suggestion', current_step, step_name, step_config['content'])
        return current_step + 1, False, training_style, thankyou_style

    return current_step, False, training_style, thankyou_style


# # state loading callback for every step
# @app.callback(
#     Output('current-step', 'data', allow_duplicate=True),
#     [Input('current-step', 'data')],
#     [State('experiment-id-store', 'data'),
#      State('visibility-mode-dropdown', 'value'),
#      State('assembly-data-store', 'data'),
#      State('clicked-buttons-store', 'data')],
#     prevent_initial_call=True
# )
# def log_step_load(current_step, experiment_id, mode, assembly_data, clicked_buttons):
#     if current_step <= 0 or current_step > len(assembly_data):
#         return current_step
#
#     step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
#     button_states = get_complete_button_states(current_step, clicked_buttons)
#
#     log_interaction(experiment_id, mode, 'step_loaded', current_step, step_name, button_states)
#     return current_step


# button state callback
@app.callback(
    [Output('short-text-btn', 'disabled', allow_duplicate=True),
     Output('long-text-btn', 'disabled', allow_duplicate=True),
     Output('single-pieces-btn', 'disabled', allow_duplicate=True),
     Output('assembly-btn', 'disabled', allow_duplicate=True),
     Output('video-btn', 'disabled', allow_duplicate=True)],
    [Input('current-step', 'data'),
     Input('enabled-interactions-store', 'data'),
     Input('clicked-buttons-store', 'data'),
     Input('short-text-content', 'style'),
     Input('long-text-content', 'style'),
     Input('single-pieces-img', 'style'),
     Input('assembly-img', 'style'),
     Input('video-player', 'style')],
    prevent_initial_call=True
)
def update_button_states(current_step, enabled_interactions, clicked_buttons,
                         short_text_style, long_text_style, single_pieces_style,
                         assembly_style, video_style):
    # Find the configuration for the current step in enabled interactions
    step_config = next((step for step in enabled_interactions['steps']
                        if step['step_id'] == current_step),
                       {'buttons': {
                           'short_text': True,
                           'long_text': True,
                           'single_pieces': True,
                           'assembly': True,
                           'video': True
                       }})

    buttons = step_config['buttons']

    # Get the clicked state for the current step using string key
    step_key = str(current_step)
    step_clicked = clicked_buttons.get(step_key, {})

    # Function to check if content is currently visible
    def is_content_visible(style):
        return style.get('display', '') == 'block'

    # A button should be disabled if:
    # 1. It's disabled in the JSON configuration, or
    # 2. It has been clicked in the current step, or
    # 3. It is initially set to be visible (preventing user interaction), or
    # 4. The content is currently visible
    return (
        not buttons.get('short_text', True) or
        step_clicked.get('short_text', False) or
        # initial_config.get('short_text', False) or
        is_content_visible(short_text_style),

        not buttons.get('long_text', True) or
        step_clicked.get('long_text', False) or
        # initial_config.get('long_text', False) or
        is_content_visible(long_text_style),

        not buttons.get('single_pieces', True) or
        step_clicked.get('single_pieces', False) or
        # initial_config.get('single_pieces', False) or
        is_content_visible(single_pieces_style),

        not buttons.get('assembly', True) or
        step_clicked.get('assembly', False) or
        # initial_config.get('assembly', False) or
        is_content_visible(assembly_style),

        not buttons.get('video', True) or
        step_clicked.get('video', False) or
        # initial_config.get('video', False) or
        is_content_visible(video_style)
    )


# Reset button labels and placeholders when changing steps
@app.callback(
    [Output('short-text-placeholder', 'style', allow_duplicate=True),
     Output('short-text-content', 'style', allow_duplicate=True),
     Output('long-text-placeholder', 'style', allow_duplicate=True),
     Output('long-text-content', 'style', allow_duplicate=True),
     Output('single-pieces-placeholder', 'style', allow_duplicate=True),
     Output('single-pieces-img', 'style', allow_duplicate=True),
     Output('assembly-placeholder', 'style', allow_duplicate=True),
     Output('assembly-img', 'style', allow_duplicate=True),
     Output('video-placeholder', 'style', allow_duplicate=True),
     Output('video-player', 'style', allow_duplicate=True),
     Output('short-text-btn', 'children', allow_duplicate=True),
     Output('long-text-btn', 'children', allow_duplicate=True),
     Output('single-pieces-btn', 'children', allow_duplicate=True),
     Output('assembly-btn', 'children', allow_duplicate=True),
     Output('video-btn', 'children', allow_duplicate=True),
     Output('video-player', 'autoPlay', allow_duplicate=True),
     Output('user-preferences-store', 'data', allow_duplicate=True)],
    [Input('current-step', 'data')],
    [State('initial-visibility-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value'),
     State('user-preferences-store', 'data'),
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def reset_button_states_and_visibility(
        current_step,
        initial_visibility,
        clicked_buttons,
        experiment_id,
        mode,
        user_preferences=None,
        assembly_data=None
):
    # Defensive programming: ensure data exists
    if not initial_visibility or 'steps' not in initial_visibility:
        # Fallback to default visibility
        initial_visibility = {
            'steps': [{
                'step_id': current_step,
                'content': {
                    'short_text': False,
                    'long_text': False,
                    'single_pieces': False,
                    'assembly': False,
                    'video': False
                }
            }]
        }

    # Find the configuration for the current step
    try:
        step_config = next(
            (step for step in initial_visibility['steps'] if step['step_id'] == current_step),
            initial_visibility['steps'][0]  # Default to first step if no match
        )
    except (IndexError, KeyError):
        # Fallback to default configuration
        step_config = {
            'content': {
                'short_text': False,
                'long_text': False,
                'single_pieces': False,
                'assembly': False,
                'video': False
            }
        }

    content = step_config.get('content', {})
    # Defensive conversion to ensure boolean
    content = {k: bool(v) for k, v in content.items()}

    # Apply adaptive logic only for dynamically adaptive mode
    if mode == 'initial_visibility_dynamically_adaptive.json' and current_step > 1:
        # Ensure user_preferences is initialized
        if user_preferences is None:
            user_preferences = {}

        # Robust step type extraction
        step_type = None
        if assembly_data and 0 < current_step - 1 < len(assembly_data):
            step_type = assembly_data[current_step - 1].get('category')

        # Fallback if step_type is not found or is None
        if step_type:
            # Ensure step_type is clean and consistent
            step_type = step_type.strip()

            # Calculate weighted frequencies
            weighted_freqs = calculate_weighted_frequencies(user_preferences)

            # Debugging print
            # print(f"Weighted frequencies for step {current_step}: {weighted_freqs}")

            # If we have preferences for this step type, use the most frequent
            if step_type in weighted_freqs and weighted_freqs[step_type]:
                # Get the most frequent content type
                most_frequent = max(
                    weighted_freqs[step_type].items(),
                    key=lambda x: x[1]
                )[0]

                # Reset all content to False
                content = {k: False for k in content}
                # Set the most frequent content to True
                content[most_frequent] = True

                step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
                log_interaction(experiment_id, mode, 'computed_suggestion', current_step, step_name, content)

    # Default labels
    default_label = [html.I(className="bi bi-eye-fill me-1"), "Show"]
    viewed_label = [html.I(className="bi bi-eye-fill me-1"), "Viewed"]

    # Get the clicked state for the current step
    step_key = str(current_step)
    step_clicked = clicked_buttons.get(step_key, {})

    # Helper function to determine visibility and label
    def get_visibility(content_type):
        is_initially_visible = content.get(content_type, False)
        is_clicked = step_clicked.get(content_type, False)

        if is_clicked:
            # If clicked, always show
            return (
                {'display': 'none'},
                {'display': 'block', **styles['image-content']},
                viewed_label
            )
        elif is_initially_visible:
            # If initially visible, but not clicked
            return (
                {'display': 'none'},
                {'display': 'block', **styles['image-content']},
                default_label
            )
        else:
            # If not initially visible and not clicked
            return (
                {'display': 'block'},
                {'display': 'none', **styles['image-content']},
                default_label
            )

    # Get visibility for each content type
    short_text_placeholder, short_text_content, short_text_btn = get_visibility('short_text')
    long_text_placeholder, long_text_content, long_text_btn = get_visibility('long_text')
    single_pieces_placeholder, single_pieces_content, single_pieces_btn = get_visibility('single_pieces')
    assembly_placeholder, assembly_content, assembly_btn = get_visibility('assembly')
    video_placeholder, video_content, video_btn = get_visibility('video')

    # Return all states
    return (
        short_text_placeholder, short_text_content,
        long_text_placeholder, long_text_content,
        single_pieces_placeholder, single_pieces_content,
        assembly_placeholder, assembly_content,
        video_placeholder, video_content,
        short_text_btn, long_text_btn,
        single_pieces_btn, assembly_btn, video_btn,
        False,  # autoPlay
        user_preferences
    )


# Restart button callback
@app.callback(
    [Output('intro-container', 'style', allow_duplicate=True),
     Output('training-container', 'style', allow_duplicate=True),
     Output('thankyou-container', 'style', allow_duplicate=True),
     Output('experiment-id-store', 'data', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True),
     Output('clicked-buttons-store', 'data', allow_duplicate=True),
     Output('user-preferences-store', 'data', allow_duplicate=True)],  # Add this output
    [Input('restart-button', 'n_clicks')],
    [State('experiment-id-store', 'data'),
     State('visibility-mode-dropdown', 'value')],
    prevent_initial_call=True
)
def restart_application(n_clicks, experiment_id, mode):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Log restart action
    log_interaction(experiment_id, mode, 'restart_application')

    # Reset to intro screen
    return (
        styles['intro-screen'],  # Show intro screen
        {'display': 'none'},  # Hide training screen
        {'display': 'none'},  # Hide thank you screen
        None,  # Reset experiment ID
        0,  # Reset step to 0
        {},  # Reset clicked buttons
        {}  # Reset user preferences
    )


# Ensure the required directories exist
def ensure_directories_exist():
    directories = [
        './images_single_pieces',
        './images_assembly',
        './images/assembly_process',
        './videos'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


@app.callback(
    Output('sentient-profile-form', 'style'),
    Input('visibility-mode-dropdown', 'value')
)
def show_profile_form(mode_value):
    return {'display': 'block'} if mode_value == 'sentient.json' else {'display': 'none'}


def validate_and_enforce_visibility(
        visibility: dict,
        step_id: int,
        enabled_interactions: dict,
        step_category: str
) -> dict:
    """
    Valida e forza i vincoli di visibilità basati su:
    1. enabled_interactions.json (vincoli tecnici per step)
    2. Regole per categoria (WITHDRAW, ASSEMBLY, CONTROL)

    Args:
        visibility: Dict con le visibilità suggerite da Gemini
        step_id: ID dello step corrente
        enabled_interactions: Configurazione da enabled_interactions.json
        step_category: Categoria dello step (withdraw, assembly, control)

    Returns:
        Dict con visibilità corrette e validate
    """

    # Trova la configurazione per questo step
    step_config = next(
        (step for step in enabled_interactions['steps'] if step['step_id'] == step_id),
        None
    )

    if not step_config:
        # Fallback: disabilita tutto se non trovato
        print(f"⚠️ No config found for step {step_id}, disabling all content")
        return {k: False for k in visibility.keys()}

    buttons = step_config.get('buttons', {})
    validated = {}

    # Regole specifiche per categoria
    category_rules = {
        'withdraw': {
            # WITHDRAW: Solo short_text e single_pieces disponibili
            'long_text': False,
            'assembly': False,
            'video': False
        },
        'control': {
            # CONTROL: NO assembly image
            'assembly': False
        },
        'assembly': {
            # ASSEMBLY: Tutti disponibili (se abilitati in buttons)
        }
    }

    category_lower = step_category.lower() if step_category else ''
    rules = category_rules.get(category_lower, {})

    for content_type, suggested_visibility in visibility.items():
        # Check 1: Il bottone è abilitato in enabled_interactions.json?
        button_enabled = buttons.get(content_type, False)

        # Check 2: La categoria lo permette?
        category_allowed = rules.get(content_type, True)  # Default: allowed

        # La visibilità finale è: suggerita AND abilitata AND permessa dalla categoria
        final_visibility = suggested_visibility and button_enabled and category_allowed

        validated[content_type] = final_visibility

        # Log se c'è stata una modifica
        if suggested_visibility and not final_visibility:
            reason = []
            if not button_enabled:
                reason.append(f"button disabled in config")
            if not category_allowed:
                reason.append(f"not allowed for {category_lower} steps")

            print(f"🚫 Overriding {content_type} visibility for step {step_id}: {', '.join(reason)}")

    return validated


def test_log_interaction():
    """Test che verifica che il logging funzioni correttamente"""
    import os

    # Crea un CSV di test
    test_csv = 'test_interaction_logs.csv'
    if os.path.exists(test_csv):
        os.remove(test_csv)

    # Backup del manager originale
    original_csv = 'interaction_logs.csv'
    import shutil
    if os.path.exists(original_csv):
        shutil.copy(original_csv, f'{original_csv}.backup')

    # Test 1: Log senza button_states
    log_interaction('test_user', 'sentient.json', 'start_experiment', 0, 'Start')

    # Test 2: Log con button_states
    log_interaction(
        'test_user',
        'sentient.json',
        'step_loaded',
        1,
        'Step 1',
        button_states={
            'short_text': True,
            'long_text': False,
            'single_pieces': True,
            'assembly': False,
            'video': False
        }
    )

    # Verifica che il CSV abbia le colonne corrette
    df = pd.read_csv('interaction_logs.csv')
    required_cols = ['experiment_id', 'mode', 'timestamp', 'action', 'step_id', 'step_name',
                     'short_text_viewed', 'long_text_viewed', 'single_pieces_viewed',
                     'assembly_viewed', 'video_viewed']

    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False

    print(f"✅ All required columns present!")
    print(f"✅ CSV has {len(df)} rows")
    print(df.head())

    # Test che HistoricalDataManager possa caricare i dati
    manager = HistoricalDataManager('interaction_logs.csv')
    print(f"✅ HistoricalDataManager loaded {len(manager.history.users)} users")

    return True


# Esegui il test solo se eseguito direttamente
if __name__ == '__main__':
    # test_log_interaction()  # Decommentare per testare
    app.run(debug=False, host='0.0.0.0', port=3000)
