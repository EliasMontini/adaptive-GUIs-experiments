import datetime
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import flask
import pandas as pd
from dash import dcc, html, Input, Output, State

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
def log_interaction(experiment_id, action, step_id=None, step_name=None, button_states=None):
    df = init_log_df()
    new_row = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        'action': action,
        'step_id': step_id if step_id is not None else 'N/A',
        'step_name': step_name if step_name is not None else 'N/A'
    }

    # Aggiungi lo stato di tutti i pulsanti se fornito
    if button_states:
        for button_name, state in button_states.items():
            new_row[f'{button_name}_viewed'] = 1 if state else 0

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('interaction_logs.csv', index=False)
    return


# Inizializzazione del DataFrame di log con colonne aggiuntive
def init_log_df():
    columns = [
        'experiment_id', 'timestamp', 'action', 'step_id', 'step_name',
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


def calculate_weighted_frequencies(
        preferences: Dict[str, Dict[str, List[float]]],
        decay_factor: float = 0.9,
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

    # Introduction page
    html.Div(id='intro-container',
             style=styles['intro-screen'],
             children=[
                 html.Div(style={'width': '400px', 'padding': '30px', 'border-radius': '8px',
                                 }, children=[
                     html.H1("Assembly Training Dashboard", style={'text-align': 'center', 'margin-bottom': '20px'}),
                     html.P("Welcome to the Assembly Training Dashboard. Please enter an experiment ID to begin.",
                            style={'margin-bottom': '20px'}),
                     dbc.Input(id='experiment-id-input', type='text', placeholder='Enter Experiment ID',
                               style={'margin-bottom': '20px'}),

                     html.P("Choose the UI mode.",
                            style={'margin-bottom': '5px'}),
                     dcc.Dropdown(
                         id='visibility-mode-dropdown',
                         options=[
                             {'label': 'Data Collection', 'value': 'initial_visibility_data_collection.json'},
                             {'label': 'Dynamically Adaptive', 'value': 'initial_visibility_dynamically_adaptive.json'},
                             {'label': 'Rule-Based Adaptive', 'value': 'initial_visibility_rule_based_adaptive.json'},
                             {'label': 'Static Mode', 'value': 'initial_visibility_static_mode.json'}
                         ],
                         value='initial_visibility_data_collection.json',  # Default selection
                         clearable=False,
                         style={'text-align': 'center', 'margin-bottom': '5px'}
                     ),
                     dbc.Button("Begin Training", id='begin-button', color='primary', style={'width': '100%'})
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
                    dbc.Button("NEXT", id='next-button', color='primary',
                               style={'padding': '20px 20px', 'width': '150px'}),
                ])
            ]),

            # Top row: Text descriptions
            html.Div(className="row mb-1", children=[
                # Short text area
                html.Div(className="col-md-4", style={'height': '150px'}, children=[
                    html.Span("Short description", className="h5 d-block mb-1"),
                    html.Div(style=styles['text-content-area'], children=[
                        html.Div(id="short-text-placeholder", className="placeholder-glow", children=[
                            html.Span(className="placeholder col-5"),
                            html.Span(className="placeholder col-3"),
                            html.Span(className="placeholder col-4"),
                            html.Span(className="placeholder col-4")
                        ]),
                        html.Div(id="short-text-content", style={'display': 'none'}),
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
                    html.Span("Long description", className="h5 d-block mb-1"),
                    html.Div(style=styles['text-content-area'], children=[
                        html.Div(id="long-text-placeholder", className="placeholder-glow", children=[
                            html.Span(className="placeholder col-7"),
                            html.Span(className="placeholder col-4"),
                            html.Span(className="placeholder col-6")
                        ]),
                        html.Div(id="long-text-content", style={'display': 'none'}),
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
                    html.Span("Image Single", className="h5 d-block mb-2"),
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
                    html.Span("Assembled parts", className="h5 d-block mb-2"),
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

    html.Div(style=styles['footer-container'], children=[
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
            html.Img(src='/assets/logosps.png', style=styles['image-style']),  # Left Image
            html.Img(src='/assets/logoxr.png', style=styles['image-style'])  # Right Image
        ]),
    ])
])


# Callbacks

# Begin button callback
@app.callback(
    [Output('intro-container', 'style'),
     Output('training-container', 'style'),
     Output('experiment-id-store', 'data'),
     Output('current-step', 'data')],
    [Input('begin-button', 'n_clicks')],
    [State('experiment-id-input', 'value')]
)
def begin_training(n_clicks, experiment_id):
    # Only proceed if button has been clicked
    if n_clicks is None:
        return styles['intro-screen'], {'display': 'none'}, None, 0

    if not experiment_id:
        experiment_id = 'unknown'

    # Log the start of the experiment
    log_interaction(experiment_id, 'start_experiment')

    # Hide intro and show training
    return {'display': 'none'}, {'display': 'block', **styles['training-screen']}, experiment_id, 1


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


# Update step content
@app.callback(
    [Output('step-header', 'children'),
     Output('step-progress-bar', 'value'),
     Output('step-text', 'children'),
     Output('experiment-id-display', 'children'),
     Output('single-pieces-img', 'src'),
     Output('assembly-img', 'src'),
     Output('video-player', 'src'),
     Output('short-text-content', 'children'),
     Output('long-text-content', 'children')],
    [Input('current-step', 'data'),
     Input('assembly-data-store', 'data'),
     Input('experiment-id-store', 'data')]
)
def update_step_content(current_step, assembly_data, experiment_id):
    if current_step <= 0 or current_step > len(assembly_data):
        return "", 0, "", "", "", "", "", "", ""

    step = assembly_data[current_step - 1]
    step_name = step['name']

    # Get content
    short_text = step['adaptive_fields']['short_text']
    long_text = step['adaptive_fields']['long_text']

    # File paths
    image_single_pieces = step['adaptive_fields']['image_single_pieces']
    image_assembly = step['adaptive_fields']['image_assembly']
    video = step['adaptive_fields']['video']

    # Calculate progress percentage
    progress_value = (current_step / len(assembly_data)) * 100
    step_text = f"Step {current_step} of {len(assembly_data)}"
    experiment_id_display = f"Experiment ID: {experiment_id}"

    return (
        f"{step_name} ",
        progress_value,
        step_text,
        experiment_id_display,
        image_single_pieces,
        image_assembly,
        video,
        short_text,
        long_text
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
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_short_text(n_clicks, placeholder_style, experiment_id, current_step,
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
        log_interaction(experiment_id, "toggle_short_text_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['short_text'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_short_text_block", current_step, step_name, button_states)

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
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_long_text(n_clicks, placeholder_style, experiment_id, current_step,
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
        log_interaction(experiment_id, "toggle_long_text_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['long_text'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_long_text_block", current_step, step_name, button_states)

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
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_single_pieces(n_clicks, placeholder_style, experiment_id, current_step,
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
        log_interaction(experiment_id, "toggle_single_pieces_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['single_pieces'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_single_pieces_block", current_step, step_name, button_states)

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
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_assembly(n_clicks, placeholder_style, experiment_id, current_step,
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
        log_interaction(experiment_id, "toggle_assembly_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences
    else:
        # Show content and update preferences
        clicked_buttons[step_key]['assembly'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_assembly_block", current_step, step_name, button_states)

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
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('user-preferences-store', 'data')],  # Added for adaptive mode
    prevent_initial_call=True
)
def toggle_video(n_clicks, placeholder_style, experiment_id, current_step,
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
        log_interaction(experiment_id, "toggle_video_none", current_step, step_name, button_states)
        return {'display': 'block'}, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons, user_preferences, False
    else:
        # Show content, update preferences, and autoplay
        clicked_buttons[step_key]['video'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_video_block", current_step, step_name, button_states)

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
    [State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('experiment-id-store', 'data'),
     State('clicked-buttons-store', 'data'),
     State('training-container', 'style'),
     State('thankyou-container', 'style')],
    prevent_initial_call=True
)
def navigate_steps(prev_clicks, next_clicks, current_step, assembly_data, experiment_id,
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
        log_interaction(experiment_id, 'navigate_previous', current_step, step_name, button_states)
        # If we're returning from thank you page to the last step
        if current_step == len(assembly_data) + 1:
            return len(assembly_data), False, {'display': 'block', **styles['training-screen']}, {'display': 'none'}
        return current_step - 1, False, training_style, thankyou_style

    # If user clicks next on the last step, show thank you page
    elif button_id == 'next-button' and current_step == len(assembly_data):
        log_interaction(experiment_id, 'navigate_to_thankyou', current_step, step_name, button_states)
        return len(assembly_data) + 1, False, {'display': 'none'}, {'display': 'block'}

    # Regular next button navigation
    elif button_id == 'next-button' and current_step < len(assembly_data):
        log_interaction(experiment_id, 'navigate_next', current_step, step_name, button_states)
        return current_step + 1, False, training_style, thankyou_style

    return current_step, False, training_style, thankyou_style


# state loading callback for every step
@app.callback(
    Output('current-step', 'data', allow_duplicate=True),
    [Input('current-step', 'data')],
    [State('experiment-id-store', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def log_step_load(current_step, experiment_id, assembly_data, clicked_buttons):
    if current_step <= 0 or current_step > len(assembly_data):
        return current_step

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
    button_states = get_complete_button_states(current_step, clicked_buttons)

    log_interaction(experiment_id, 'step_loaded', current_step, step_name, button_states)
    return current_step


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
     Input('initial-visibility-store', 'data'),
     Input('short-text-content', 'style'),
     Input('long-text-content', 'style'),
     Input('single-pieces-img', 'style'),
     Input('assembly-img', 'style'),
     Input('video-player', 'style')],
    prevent_initial_call=True
)
def update_button_states(current_step, enabled_interactions, clicked_buttons, initial_visibility,
                         short_text_style, long_text_style, single_pieces_style,
                         assembly_style, video_style):
    # Find the configuration for the current step in initial visibility
    initial_config = next((step['content'] for step in initial_visibility['steps']
                           if step['step_id'] == current_step),
                          {'short_text': False, 'long_text': False,
                           'single_pieces': False, 'assembly': False, 'video': False})

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
     State('visibility-mode-dropdown', 'value'),
     State('user-preferences-store', 'data'),
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def reset_button_states_and_visibility(
        current_step,
        initial_visibility,
        clicked_buttons,
        visibility_mode=None,
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
    if visibility_mode == 'initial_visibility_dynamically_adaptive.json' and current_step > 1:
        # Ensure user_preferences is initialized
        if user_preferences is None:
            user_preferences = {}

        # Robust step type extraction
        step_type = None
        if assembly_data and 0 < current_step - 1 < len(assembly_data):
            step_type = assembly_data[current_step - 1].get('category')

        # Fallback if step_type is not found or is None
        if not step_type:
            print(f"Warning: Could not determine step type for step {current_step}")
            return None  # or handle as needed

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

        # Track the initially visible content
        initially_visible_content = [k for k, v in content.items() if v]
        user_preferences = update_user_preferences(
            user_preferences,
            step_type,  # Use the correctly extracted step type
            initially_visible_content,
            datetime.now().timestamp(),
            is_initial=True
        )

        print(f"Updated user preferences at step {current_step} (Type: {step_type}):", user_preferences)

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
    [State('experiment-id-store', 'data')],
    prevent_initial_call=True
)
def restart_application(n_clicks, experiment_id):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Log restart action
    log_interaction(experiment_id, 'restart_application')

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


# Run the app
if __name__ == '__main__':
    ensure_directories_exist()
    app.run_server(debug=False, host='0.0.0.0', port=3000)
