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
def log_interaction(experiment_id, action, step_id=None, step_name=None):
    df = init_log_df()
    new_row = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        'action': action,
        'step_id': step_id if step_id is not None else 'N/A',
        'step_name': step_name if step_name is not None else 'N/A'
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('interaction_logs.csv', index=False)
    return


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
        'overflow-y': 'hidden'
    },
    'content-container': {
        'margin-bottom': '20px',
        'height': '100%'
    },
    'button-container': {
        'text-align': 'center',
        'margin-top': '10px'
    },
    'image-container': {
        'display': 'flex',
        'flex-direction': 'column',
        'align-items': 'center',
        'justify-content': 'space-between',
        'height': '100%',
        'min-height': '200px',  # Fixed height to prevent layout shifts
    },
    'image-wrapper': {
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'width': '100%',
        'flex-grow': 1,
    },
    'footer-container': {
        'display': 'flex',
        'justify-content': 'space-between',
        'padding': '10px 0',
        'margin-top': '20px',
        'border-top': '1px solid #dee2e6'
    }
}

# App layout
app.layout = html.Div([
    # Store components for state management
    dcc.Store(id='current-step', data=0),  # 0 = intro, 1+ = steps
    dcc.Store(id='experiment-id-store', data=None),
    dcc.Store(id='assembly-data-store', data=load_assembly_process()),
    dcc.Store(id='navigation-in-progress', data=False),  # Store to track navigation state

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
            html.H1("Adaptive UI", className="mb-4 mt-3"),

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
                    html.Div(id="short-text-placeholder", className="placeholder-glow", children=[
                        html.Span(className="placeholder col-5"),
                        html.Span(className="placeholder col-3"),
                        html.Span(className="placeholder col-4")
                    ]),
                    html.Div(id="short-text-content", style={'display': 'none'}),
                    html.Div(className="mt-2", children=[
                        dbc.Button([
                            html.I(className="bi bi-eye-fill me-1"),
                            "Show"
                        ], id="short-text-btn", color="primary")
                    ])
                ]),

                # Long text area
                html.Div(className="col-md-8", children=[
                    html.Span("Long description", className="h5 d-block mb-2"),
                    html.Div(id="long-text-placeholder", className="placeholder-glow", children=[
                        html.Span(className="placeholder col-7"),
                        html.Span(className="placeholder col-4"),
                        html.Span(className="placeholder col-6")
                    ]),
                    html.Div(id="long-text-content", style={'display': 'none'}),
                    html.Div(className="mt-2", children=[
                        dbc.Button([
                            html.I(className="bi bi-eye-fill me-1"),
                            "Show"
                        ], id="long-text-btn", color="primary")
                    ])
                ])
            ]),

            # Bottom row: Visual content
            html.Div(className="row", children=[
                # Individual Parts image
                html.Div(className="col-md-4", children=[
                    html.Div(style=styles['image-container'], children=[
                        html.Span("Image", className="h5 d-block mb-2"),
                        html.Div(style=styles['image-wrapper'], children=[
                            html.Img(id="single-pieces-placeholder",
                                     src=placeholder_img,
                                     className="img-fluid")
                        ]),
                        html.Img(id="single-pieces-img",
                                 style={'display': 'none', 'max-width': '100%', 'max-height': '300px'},
                                 className="img-fluid"),
                        html.Div(className="mt-2 mb-3 w-100 text-center", children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="single-pieces-btn", color="primary")
                        ])
                    ])
                ]),

                # Assembled Parts image
                html.Div(className="col-md-4", children=[
                    html.Div(style=styles['image-container'], children=[
                        html.Span("Assembled parts", className="h5 d-block mb-2"),
                        html.Div(style=styles['image-wrapper'], children=[
                            html.Img(id="assembly-placeholder",
                                     src=placeholder_img,
                                     className="img-fluid")
                        ]),
                        html.Img(id="assembly-img",
                                 style={'display': 'none', 'max-width': '100%', 'max-height': '300px'},
                                 className="img-fluid"),
                        html.Div(className="mt-2 mb-3 w-100 text-center", children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="assembly-btn", color="primary")
                        ])
                    ])
                ]),
                # Replace the entire video column section with this updated version
                html.Div(className="col-md-4", children=[
                    html.Div(style=styles['image-container'], children=[
                        html.Span("Video", className="h5 d-block mb-2"),
                        # Image wrapper div
                        html.Div(style=styles['image-wrapper'], children=[
                            # Placeholder - structured more like the image placeholders
                            html.Img(id="video-placeholder",
                                     src=placeholder_img,
                                     className="img-fluid")
                        ]),
                        # Video player
                        html.Video(id="video-player",
                                   controls=True,
                                   style={'display': 'none', 'max-width': '100%', 'max-height': '300px'},
                                   className="img-fluid mb-3"),
                        # Button container
                        html.Div(className="mt-2 mb-3 w-100 text-center", children=[
                            dbc.Button([
                                html.I(className="bi bi-eye-fill me-1"),
                                "Show"
                            ], id="video-btn", color="primary", size="lg", className="px-4")
                        ])
                    ])
                ])
            ]),

            # Footer
            html.Div(style=styles['footer-container'], children=[
                html.Div(id='step-counter'),
                html.Div(id='experiment-id-display')
            ])
        ])
    ])
])


# Callbacks

# Begin button
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


# Previous and Next buttons
@app.callback(
    [Output('current-step', 'data', allow_duplicate=True),
     Output('navigation-in-progress', 'data', allow_duplicate=True)],
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('current-step', 'data'),
     State('assembly-data-store', 'data'),
     State('experiment-id-store', 'data')],
    prevent_initial_call=True
)
def navigate_steps(prev_clicks, next_clicks, current_step, assembly_data, experiment_id):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_step, False

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    if button_id == 'prev-button' and current_step > 1:
        log_interaction(experiment_id, 'navigate_previous', current_step, step_name)
        return current_step - 1, False
    elif button_id == 'next-button' and current_step < len(assembly_data):
        log_interaction(experiment_id, 'navigate_next', current_step, step_name)
        return current_step + 1, False
    return current_step, False


# Update step content
@app.callback(
    [Output('step-header', 'children'),
     Output('step-counter', 'children'),
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
        return "", "", "", "", "", "", "", ""

    step = assembly_data[current_step - 1]
    step_name = step['name']

    # Get content
    short_text = step['adaptive_fields']['short_text']
    long_text = step['adaptive_fields']['long_text']

    # File paths
    image_single_pieces = step['adaptive_fields']['image_single_pieces']
    image_assembly = step['adaptive_fields']['image_assembly']
    video = step['adaptive_fields']['video']

    step_counter = f"Step {current_step} of {len(assembly_data)}"
    experiment_id_display = f"Experiment ID: {experiment_id}"

    return (
        step_name,
        step_counter,
        experiment_id_display,
        image_single_pieces,
        image_assembly,
        video,
        short_text,
        long_text
    )


# Reset visibility when navigating
@app.callback(
    [Output('short-text-btn', 'color'),
     Output('long-text-btn', 'color'),
     Output('single-pieces-btn', 'color'),
     Output('assembly-btn', 'color'),
     Output('video-btn', 'color'),
     Output('short-text-placeholder', 'style'),
     Output('long-text-content', 'style'),
     Output('single-pieces-placeholder', 'style'),
     Output('single-pieces-img', 'style'),
     Output('assembly-placeholder', 'style'),
     Output('assembly-img', 'style'),
     Output('video-placeholder', 'style'),
     Output('video-player', 'style'),
     Output('long-text-placeholder', 'style'),
     Output('short-text-content', 'style')],
    [Input('current-step', 'data')],
    prevent_initial_call=True
)
def reset_content_visibility(current_step):
    # Reset all buttons to primary and show placeholders, hide content
    placeholder_visible = {'display': 'block'}
    content_hidden = {'display': 'none'}
    image_content_hidden = {'display': 'none', 'max-width': '100%', 'max-height': '300px'}

    return ('primary', 'primary', 'primary', 'primary', 'primary',
            placeholder_visible, content_hidden,
            placeholder_visible, image_content_hidden,
            placeholder_visible, image_content_hidden,
            placeholder_visible, image_content_hidden,
            placeholder_visible, content_hidden)


# Toggle short text
@app.callback(
    [Output('short-text-placeholder', 'style', allow_duplicate=True),
     Output('short-text-content', 'style', allow_duplicate=True),
     Output('short-text-btn', 'children')],
    [Input('short-text-btn', 'n_clicks')],
    [State('short-text-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def toggle_short_text(n_clicks, placeholder_style, experiment_id, current_step, assembly_data):
    if n_clicks is None:
        return placeholder_style, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"), "Show"]

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    is_showing = placeholder_style.get('display') == 'none'

    if is_showing:
        # Hide content
        log_interaction(experiment_id, "toggle_short_text_none", current_step, step_name)
        return {'display': 'block'}, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"), "Show"]
    else:
        # Show content
        log_interaction(experiment_id, "toggle_short_text_block", current_step, step_name)
        return {'display': 'none'}, {'display': 'block'}, [html.I(className="bi bi-eye-slash me-1"), "Hide"]


# Toggle long text
@app.callback(
    [Output('long-text-placeholder', 'style', allow_duplicate=True),
     Output('long-text-content', 'style', allow_duplicate=True),
     Output('long-text-btn', 'children')],
    [Input('long-text-btn', 'n_clicks')],
    [State('long-text-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def toggle_long_text(n_clicks, placeholder_style, experiment_id, current_step, assembly_data):
    if n_clicks is None:
        return placeholder_style, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"), "Show"]

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    is_showing = placeholder_style.get('display') == 'none'

    if is_showing:
        # Hide content
        log_interaction(experiment_id, "toggle_long_text_none", current_step, step_name)
        return {'display': 'block'}, {'display': 'none'}, [html.I(className="bi bi-eye-fill me-1"), "Show"]
    else:
        # Show content
        log_interaction(experiment_id, "toggle_long_text_block", current_step, step_name)
        return {'display': 'none'}, {'display': 'block'}, [html.I(className="bi bi-eye-slash me-1"), "Hide"]


# Toggle single pieces image
@app.callback(
    [Output('single-pieces-placeholder', 'style', allow_duplicate=True),
     Output('single-pieces-img', 'style', allow_duplicate=True),
     Output('single-pieces-btn', 'children')],
    [Input('single-pieces-btn', 'n_clicks')],
    [State('single-pieces-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def toggle_single_pieces(n_clicks, placeholder_style, experiment_id, current_step, assembly_data):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', 'max-width': '100%', 'max-height': '300px'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"]

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    is_showing = placeholder_style.get('display') == 'none'

    if is_showing:
        # Hide content
        log_interaction(experiment_id, "toggle_single_pieces_none", current_step, step_name)
        return {'display': 'block'}, {'display': 'none', 'max-width': '100%', 'max-height': '300px'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"]
    else:
        # Show content
        log_interaction(experiment_id, "toggle_single_pieces_block", current_step, step_name)
        return {'display': 'none'}, {'display': 'block', 'max-width': '100%', 'max-height': '300px'}, [
            html.I(className="bi bi-eye-slash me-1"), "Hide"]


# Toggle assembly image
@app.callback(
    [Output('assembly-placeholder', 'style', allow_duplicate=True),
     Output('assembly-img', 'style', allow_duplicate=True),
     Output('assembly-btn', 'children')],
    [Input('assembly-btn', 'n_clicks')],
    [State('assembly-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def toggle_assembly(n_clicks, placeholder_style, experiment_id, current_step, assembly_data):
    if n_clicks is None:
        return placeholder_style, {'display': 'none', 'max-width': '100%', 'max-height': '300px'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"]

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    is_showing = placeholder_style.get('display') == 'none'

    if is_showing:
        # Hide content
        log_interaction(experiment_id, "toggle_assembly_none", current_step, step_name)
        return {'display': 'block'}, {'display': 'none', 'max-width': '100%', 'max-height': '300px'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"]
    else:
        # Show content
        log_interaction(experiment_id, "toggle_assembly_block", current_step, step_name)
        return {'display': 'none'}, {'display': 'block', 'max-width': '100%', 'max-height': '300px'}, [
            html.I(className="bi bi-eye-slash me-1"), "Hide"]


# Toggle video
# Modified toggle_video callback
@app.callback(
    [Output('video-placeholder', 'style', allow_duplicate=True),
     Output('video-player', 'style', allow_duplicate=True),
     Output('video-btn', 'children')],
    [Input('video-btn', 'n_clicks')],
    [State('video-placeholder', 'style'),
     State('experiment-id-store', 'data'),
     State('current-step', 'data'),
     State('assembly-data-store', 'data')],
    prevent_initial_call=True
)
def toggle_video(n_clicks, placeholder_style, experiment_id, current_step, assembly_data):
    if n_clicks is None:
        return {'display': 'block'}, {'display': 'none', 'max-width': '100%', 'max-height': '300px'}, [
            html.I(className="bi bi-eye-fill me-1"), "Show"]

    step_name = assembly_data[current_step - 1]['name'] if 0 < current_step <= len(assembly_data) else 'N/A'

    # Check if the button was just clicked (not based on current visibility state)
    # This ensures the toggle happens reliably
    if dash.callback_context.triggered:
        button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'video-btn':
            # If the video is currently showing (placeholder is hidden)
            current_display = placeholder_style.get('display', 'block')
            if current_display == 'none':
                # Hide video, show placeholder
                log_interaction(experiment_id, "toggle_video_none", current_step, step_name)
                return {'display': 'block'}, {'display': 'none', 'max-width': '100%', 'max-height': '300px'}, [
                    html.I(className="bi bi-eye-fill me-1"), "Show"]
            else:
                # Show video, hide placeholder
                log_interaction(experiment_id, "toggle_video_block", current_step, step_name)
                return {'display': 'none'}, {'display': 'block', 'max-width': '100%', 'max-height': '300px'}, [
                    html.I(className="bi bi-eye-slash me-1"), "Hide"]

    # Default fallback (shouldn't typically reach here)
    return placeholder_style, {'display': 'none', 'max-width': '100%', 'max-height': '300px'}, [
        html.I(className="bi bi-eye-fill me-1"), "Show"]


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
