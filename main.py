import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import json
import os
import pandas as pd
from datetime import datetime
import flask

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
        with open('settings/initial_visibility.json', 'r') as f:
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


# Define placeholder image URL
placeholder_img = "https://developers.elementor.com/docs/assets/img/elementor-placeholder-image.png"

# Define custom styles to add to the layout
styles = {
    'full-view': {
        'width': '100%',
        'height': '100vh',
        'overflow': 'hidden'
    },
    'intro-screen': {
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'background-color': '#f8f9fa',
        'height': '100vh'
    },
    'training-screen': {
        'padding': '20px',
        'height': '100vh',
        'overflow-y': 'auto'
    },
    'content-container': {
        'margin-bottom': '20px',
        'height': '100%'
    },
    'button-container': {
        'position': 'absolute',
        'bottom': '0',
        'left': '0',
        'right': '0',
        'padding': '10px',
        'background-color': 'rgba(255, 255, 255, 0.9)',
        'z-index': '10'
    },
    'text-content-area': {
        'min-height': '150px',
        'position': 'relative',
        'padding-bottom': '60px',
        'overflow': 'auto'
    },
    'image-container': {
        'position': 'relative',
        'height': '350px',
        'display': 'flex',
        'flex-direction': 'column'
    },
    'image-wrapper': {
        'flex-grow': '1',
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'overflow': 'hidden',
        'position': 'relative'
    },
    'image-content': {
        'max-width': '100%',
        'max-height': '250px',
        'object-fit': 'contain'
    },
    'footer-container': {
        'position': 'fixed',
        'bottom': '0',
        'width': '100%',
        'padding': '50px',
        'backgroundColor': '#f8f9fa',
        'textAlign': 'center'
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
        'marginTop': '10px'
    },
    'image-style': {
        'width': '100px',  # Adjust size as needed
        'height': 'auto'
    }
}

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

    # Introduction page
    html.Div(id='intro-container',
             style=styles['intro-screen'],
             children=[
                 html.Div(style={'width': '400px', 'padding': '30px', 'background': 'white', 'border-radius': '8px',
                                 'box-shadow': '0 0 15px rgba(0,0,0,0.1)'}, children=[
                     html.H1("Assembly Training Dashboard", style={'text-align': 'center', 'margin-bottom': '20px'}),
                     html.P("Welcome to the Assembly Training Dashboard. Please enter an experiment ID to begin.",
                            style={'margin-bottom': '20px'}),
                     dbc.Input(id='experiment-id-input', type='text', placeholder='Enter Experiment ID',
                               style={'margin-bottom': '20px'}),
                     dbc.Button("Begin Training", id='begin-button', color='primary', style={'width': '100%'})
                 ])
             ]),

    # Training steps page
    html.Div(id='training-container', style={'display': 'none'}, children=[
        html.Div(className="container-fluid", children=[
            # Header
            html.H1("Operator Assembly Support System", className="font-bold mb-4 mt-3"),
            html.Hr(),
            # Navigation buttons
            html.Div(className="d-flex justify-content-between align-items-center mb-3", children=[
                html.H3(id='step-header', className="m-0"),
                html.Div(className="d-flex gap-2", children=[
                    dbc.Button("Previous", id='prev-button', color='secondary'),
                    dbc.Button("Next", id='next-button', color='primary')
                ])
            ]),

            # Top row: Text descriptions
            html.Div(className="row mb-4", children=[
                # Short text area
                html.Div(className="col-md-4", children=[
                    html.Span("Short description", className="h5 d-block mb-2"),
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
                    html.Span("Long description", className="h5 d-block mb-2"),
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
                    html.Img(src='/assets/logoxr.jpg', style=styles['image-style'])  # Right Image
                ]),
            ])
        ])
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
    step_category = step['category']

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
        f"{step_name} ({step_category}) ",
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
     Output('clicked-buttons-store', 'data', allow_duplicate=True)],
    [Input('short-text-btn', 'n_clicks')],
    [State('short-text-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def toggle_short_text(n_clicks, placeholder_style, experiment_id, current_step, assembly_data, clicked_buttons):
    if n_clicks is None:
        return placeholder_style, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"),
                                                        "Show"], clicked_buttons

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
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
        return {'display': 'block'}, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"),
                                                           "Show"], clicked_buttons
    else:
        # Show content
        clicked_buttons[step_key]['short_text'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_short_text_block", current_step, step_name, button_states)
        return {'display': 'none'}, {'display': 'block'}, [html.I(className="bi bi-eye-fill me-1"),
                                                           "Viewed"], clicked_buttons


# toggle long_text
@app.callback(
    [Output('long-text-placeholder', 'style', allow_duplicate=True),
     Output('long-text-content', 'style', allow_duplicate=True),
     Output('long-text-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True)],
    [Input('long-text-btn', 'n_clicks')],
    [State('long-text-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def toggle_long_text(n_clicks, placeholder_style, experiment_id, current_step, assembly_data, clicked_buttons):
    if n_clicks is None:
        return placeholder_style, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"),
                                                        "Show"], clicked_buttons

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
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
        return {'display': 'block'}, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"),
                                                           "Show"], clicked_buttons
    else:
        # Show content
        clicked_buttons[step_key]['long_text'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_long_text_block", current_step, step_name, button_states)
        return {'display': 'none'}, {'display': 'block'}, [html.I(className="bi bi-eye-fill me-1"),
                                                           "Viewed"], clicked_buttons


# toggle single_pieces callback
@app.callback(
    [Output('single-pieces-placeholder', 'style', allow_duplicate=True),
     Output('single-pieces-img', 'style', allow_duplicate=True),
     Output('single-pieces-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True)],
    [Input('single-pieces-btn', 'n_clicks')],
    [State('single-pieces-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def toggle_single_pieces(n_clicks, placeholder_style, experiment_id, current_step, assembly_data, clicked_buttons):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
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
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons
    else:
        # Show content
        clicked_buttons[step_key]['single_pieces'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_single_pieces_block", current_step, step_name, button_states)
        return {'display': 'none'}, {'display': 'block', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons


# toggle assembly callback
@app.callback(
    [Output('assembly-placeholder', 'style', allow_duplicate=True),
     Output('assembly-img', 'style', allow_duplicate=True),
     Output('assembly-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True)],
    [Input('assembly-btn', 'n_clicks')],
    [State('assembly-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def toggle_assembly(n_clicks, placeholder_style, experiment_id, current_step, assembly_data, clicked_buttons):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
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
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons
    else:
        # Show content
        clicked_buttons[step_key]['assembly'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_assembly_block", current_step, step_name, button_states)
        return {'display': 'none'}, {'display': 'block', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons


# toggle video callback
@app.callback(
    [Output('video-placeholder', 'style', allow_duplicate=True),
     Output('video-player', 'style', allow_duplicate=True),
     Output('video-btn', 'children'),
     Output('clicked-buttons-store', 'data', allow_duplicate=True)],
    [Input('video-btn', 'n_clicks')],
    [State('video-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def toggle_video(n_clicks, placeholder_style, experiment_id, current_step, assembly_data, clicked_buttons):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'
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
            html.I(className="bi bi-eye-fill me-1"), "Show"], clicked_buttons
    else:
        # Show content
        clicked_buttons[step_key]['video'] = True
        button_states = get_complete_button_states(current_step, clicked_buttons)
        log_interaction(experiment_id, "toggle_video_block", current_step, step_name, button_states)
        return {'display': 'none'}, {'display': 'block', **styles['image-content']}, [
            html.I(className="bi bi-eye-fill me-1"), "Viewed"], clicked_buttons


# navigation callback
@app.callback(
    [Output('current-step', 'data', allow_duplicate=True),
     Output('navigation-in-progress', 'data', allow_duplicate=True)],
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('experiment-id-store', 'data'),
     State('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def navigate_steps(prev_clicks, next_clicks, current_step, assembly_data, experiment_id, clicked_buttons):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_step, False

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    # Get the current state of all buttons for logging
    button_states = get_complete_button_states(current_step, clicked_buttons)

    if button_id == 'prev-button' and current_step > 1:
        log_interaction(experiment_id, 'navigate_previous', current_step, step_name, button_states)
        return current_step - 1, False
    elif button_id == 'next-button' and current_step < len(assembly_data):
        log_interaction(experiment_id, 'navigate_next', current_step, step_name, button_states)
        return current_step + 1, False
    return current_step, False


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
     Input('clicked-buttons-store', 'data')],
    prevent_initial_call=True
)
def update_button_states(current_step, enabled_interactions, clicked_buttons):
    # Find the configuration for the current step
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

    # A button should be disabled if either:
    # 1. It's disabled in the JSON configuration, or
    # 2. It has been clicked in the current step
    return (
        not buttons.get('short_text', True) or step_clicked.get('short_text', False),
        not buttons.get('long_text', True) or step_clicked.get('long_text', False),
        not buttons.get('single_pieces', True) or step_clicked.get('single_pieces', False),
        not buttons.get('assembly', True) or step_clicked.get('assembly', False),
        not buttons.get('video', True) or step_clicked.get('video', False)
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
     Output('video-btn', 'children', allow_duplicate=True)],
    [Input('current-step', 'data')],
    [State('initial-visibility-store', 'data')],
    prevent_initial_call=True
)
def reset_button_states_and_visibility(current_step, initial_visibility):
    # Default button label
    default_label = [html.I(className="bi bi-eye-fill me-1"), "Show"]

    # Find the configuration for the current step
    step_config = next((step for step in initial_visibility['steps']
                        if step['step_id'] == current_step),
                       {'content': {
                           'short_text': False,
                           'long_text': False,
                           'single_pieces': False,
                           'assembly': False,
                           'video': False
                       }})

    content = step_config['content']

    # Define styles for visible and hidden states
    visible_placeholder = {'display': 'none'}
    hidden_placeholder = {'display': 'block'}

    # Keep the fixed position and dimensions for content
    visible_content = {'display': 'block', **styles['image-content']}
    hidden_content = {'display': 'none', **styles['image-content']}

    # For text content, we need to handle separately
    visible_text = {'display': 'block'}
    hidden_text = {'display': 'none'}

    return (
        visible_placeholder if content.get('short_text', False) else hidden_placeholder,
        visible_text if content.get('short_text', False) else hidden_text,
        visible_placeholder if content.get('long_text', False) else hidden_placeholder,
        visible_text if content.get('long_text', False) else hidden_text,
        visible_placeholder if content.get('single_pieces', False) else hidden_placeholder,
        visible_content if content.get('single_pieces', False) else hidden_content,
        visible_placeholder if content.get('assembly', False) else hidden_placeholder,
        visible_content if content.get('assembly', False) else hidden_content,
        visible_placeholder if content.get('video', False) else hidden_placeholder,
        visible_content if content.get('video', False) else hidden_content,
        default_label,
        default_label,
        default_label,
        default_label,
        default_label
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
