import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import json
import os
import pandas as pd
from datetime import datetime
import base64
import flask

# Initialize the app with Flask server to handle static files
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP],
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
        return pd.DataFrame(columns=['experiment_id', 'timestamp', 'action', 'step_id'])


# Log user interaction
def log_interaction(experiment_id, action, step_id=None):
    df = init_log_df()
    new_row = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        'action': action,
        'step_id': step_id if step_id is not None else 'N/A'
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('interaction_logs.csv', index=False)
    return


# App layout
app.layout = html.Div([
    # Store components for state management
    dcc.Store(id='current-step', data=0),  # 0 = intro, 1+ = steps
    dcc.Store(id='experiment-id-store', data=None),
    dcc.Store(id='assembly-data-store', data=load_assembly_process()),

    # Introduction page
    html.Div(id='intro-container',
             className='container-fluid vh-100 d-flex flex-column justify-content-center align-items-center', children=[
            html.Div(className='text-center', children=[
                html.H1("Assembly Training Dashboard", className='display-4 mb-4'),
                html.P("Welcome to the Assembly Training Dashboard. Please enter an experiment ID to begin.",
                       className='lead mb-4'),
                dbc.Input(id='experiment-id-input', type='text', placeholder='Enter Experiment ID',
                          className='mb-3 text-center'),
                dbc.Button("Begin Training", id='begin-button', color='primary', className='mt-3')
            ])
        ]),

    # Training steps page
    html.Div(id='training-container', className='container-fluid vh-100 p-4', style={'display': 'none'}, children=[
        dbc.Row([
            # Header
            dbc.Col([
                html.Div(className='d-flex justify-content-between align-items-center mb-4', children=[
                    html.H2(id='step-header', className='m-0'),
                    html.Div(className='d-flex', children=[
                        dbc.Button("Previous", id='prev-button', color='secondary', className='me-2'),
                        dbc.Button("Next", id='next-button', color='primary')
                    ])
                ])
            ], width=12)
        ]),

        dbc.Row([
            # Left side: Controls and text
            dbc.Col([
                html.Div(className='card mb-3', children=[
                    html.Div(className='card-header bg-primary text-white', children=[
                        html.H5("Controls", className='m-0')
                    ]),
                    html.Div(className='card-body', children=[
                        html.Div(className='d-grid gap-2', children=[
                            dbc.Button("Show Short Description", id='short-text-btn', color='outline-primary',
                                       className='text-start'),
                            dbc.Button("Show Detailed Instructions", id='long-text-btn', color='outline-primary',
                                       className='text-start'),
                            dbc.Button("Show Individual Parts", id='single-pieces-btn', color='outline-primary',
                                       className='text-start'),
                            dbc.Button("Show Assembled Part", id='assembly-btn', color='outline-primary',
                                       className='text-start'),
                            dbc.Button("Show Assembly Process", id='assembly-process-btn', color='outline-primary',
                                       className='text-start'),
                            dbc.Button("Show Video Guide", id='video-btn', color='outline-primary',
                                       className='text-start')
                        ])
                    ])
                ]),

                # Text content cards
                html.Div(id='short-text-container', className='card mb-3', style={'display': 'none'}, children=[
                    html.Div(className='card-header bg-info text-white', children=[
                        html.H5("Short Description", className='m-0')
                    ]),
                    html.Div(id='short-text-content', className='card-body')
                ]),

                html.Div(id='long-text-container', className='card mb-3', style={'display': 'none'}, children=[
                    html.Div(className='card-header bg-info text-white', children=[
                        html.H5("Detailed Instructions", className='m-0')
                    ]),
                    html.Div(id='long-text-content', className='card-body')
                ])
            ], width=4),

            # Right side: Visual content
            dbc.Col([
                # Images
                html.Div(id='single-pieces-container', className='card mb-3', style={'display': 'none'}, children=[
                    html.Div(className='card-header bg-success text-white', children=[
                        html.H5("Individual Parts", className='m-0')
                    ]),
                    html.Div(className='card-body text-center', children=[
                        html.Img(id='single-pieces-img', className='img-fluid')
                    ])
                ]),

                html.Div(id='assembly-container', className='card mb-3', style={'display': 'none'}, children=[
                    html.Div(className='card-header bg-success text-white', children=[
                        html.H5("Assembled Part", className='m-0')
                    ]),
                    html.Div(className='card-body text-center', children=[
                        html.Img(id='assembly-img', className='img-fluid')
                    ])
                ]),

                html.Div(id='assembly-process-container', className='card mb-3', style={'display': 'none'}, children=[
                    html.Div(className='card-header bg-success text-white', children=[
                        html.H5("Assembly Process", className='m-0')
                    ]),
                    html.Div(className='card-body text-center', children=[
                        html.Img(id='assembly-process-img', className='img-fluid')
                    ])
                ]),

                # Video
                html.Div(id='video-container', className='card', style={'display': 'none'}, children=[
                    html.Div(className='card-header bg-danger text-white', children=[
                        html.H5("Video Guide", className='m-0')
                    ]),
                    html.Div(className='card-body text-center', children=[
                        html.Video(id='video-player', controls=True, className='w-100')
                    ])
                ])
            ], width=8)
        ]),

        # Footer with status
        dbc.Row([
            dbc.Col([
                html.Div(className='d-flex justify-content-between align-items-center mt-3', children=[
                    html.Div(id='step-counter', className='text-muted'),
                    html.Div(id='experiment-id-display', className='text-muted')
                ])
            ], width=12)
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
    if n_clicks is None:
        return {'display': 'flex'}, {'display': 'none'}, None, 0

    if not experiment_id:
        experiment_id = 'unknown'

    # Log the start of the experiment
    log_interaction(experiment_id, 'start_experiment')

    return {'display': 'none'}, {'display': 'block'}, experiment_id, 1


# Previous and Next buttons
@app.callback(
    Output('current-step', 'data', allow_duplicate=True),
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
        return current_step

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'prev-button' and current_step > 1:
        log_interaction(experiment_id, 'navigate_previous', current_step)
        return current_step - 1
    elif button_id == 'next-button' and current_step < len(assembly_data):
        log_interaction(experiment_id, 'navigate_next', current_step)
        return current_step + 1
    return current_step


# Update step content
@app.callback(
    [Output('step-header', 'children'),
     Output('step-counter', 'children'),
     Output('experiment-id-display', 'children'),
     Output('short-text-content', 'children'),
     Output('long-text-content', 'children'),
     Output('single-pieces-img', 'src'),
     Output('assembly-img', 'src'),
     Output('assembly-process-img', 'src'),
     Output('video-player', 'src')],
    [Input('current-step', 'data'),
     Input('assembly-data-store', 'data'),
     Input('experiment-id-store', 'data')]
)
def update_step_content(current_step, assembly_data, experiment_id):
    if current_step <= 0 or current_step > len(assembly_data):
        return "", "", "", "", "", "", "", "", ""

    step = assembly_data[current_step - 1]
    step_name = step['name']

    # Get content
    short_text = step['adaptive_fields']['short_text']
    long_text = step['adaptive_fields']['long_text']

    # Since we're using the actual file paths from JSON, we can use them directly
    # The Flask routes we defined will handle serving these files
    image_single_pieces = step['adaptive_fields']['image_single_pieces']
    image_assembly = step['adaptive_fields']['image_assembly']
    image_assembly_process = step['adaptive_fields']['image_assembly_process']
    video = step['adaptive_fields']['video']

    step_counter = f"Step {current_step} of {len(assembly_data)}"
    experiment_id_display = f"Experiment ID: {experiment_id}"

    return (
        step_name,
        step_counter,
        experiment_id_display,
        short_text,
        long_text,
        image_single_pieces,
        image_assembly,
        image_assembly_process,
        video
    )


# Toggle content visibility and log interactions
@app.callback(
    [Output('short-text-container', 'style'),
     Output('long-text-container', 'style'),
     Output('single-pieces-container', 'style'),
     Output('assembly-container', 'style'),
     Output('assembly-process-container', 'style'),
     Output('video-container', 'style')],
    [Input('short-text-btn', 'n_clicks'),
     Input('long-text-btn', 'n_clicks'),
     Input('single-pieces-btn', 'n_clicks'),
     Input('assembly-btn', 'n_clicks'),
     Input('assembly-process-btn', 'n_clicks'),
     Input('video-btn', 'n_clicks')],
    [State('short-text-container', 'style'),
     State('long-text-container', 'style'),
     State('single-pieces-container', 'style'),
     State('assembly-container', 'style'),
     State('assembly-process-container', 'style'),
     State('video-container', 'style'),
     State('current-step', 'data'),
     State('experiment-id-store', 'data')]
)
def toggle_content(
        short_text_clicks, long_text_clicks, single_pieces_clicks,
        assembly_clicks, assembly_process_clicks, video_clicks,
        short_text_style, long_text_style, single_pieces_style,
        assembly_style, assembly_process_style, video_style,
        current_step, experiment_id
):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            short_text_style, long_text_style, single_pieces_style,
            assembly_style, assembly_process_style, video_style
        )

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize styles if None
    if short_text_style is None:
        short_text_style = {'display': 'none'}
    if long_text_style is None:
        long_text_style = {'display': 'none'}
    if single_pieces_style is None:
        single_pieces_style = {'display': 'none'}
    if assembly_style is None:
        assembly_style = {'display': 'none'}
    if assembly_process_style is None:
        assembly_process_style = {'display': 'none'}
    if video_style is None:
        video_style = {'display': 'none'}

    # Toggle visibility and log interaction
    if button_id == 'short-text-btn':
        display = 'block' if short_text_style.get('display') == 'none' else 'none'
        short_text_style = {'display': display}
        log_interaction(experiment_id, f"toggle_short_text_{display}", current_step)
    elif button_id == 'long-text-btn':
        display = 'block' if long_text_style.get('display') == 'none' else 'none'
        long_text_style = {'display': display}
        log_interaction(experiment_id, f"toggle_long_text_{display}", current_step)
    elif button_id == 'single-pieces-btn':
        display = 'block' if single_pieces_style.get('display') == 'none' else 'none'
        single_pieces_style = {'display': display}
        log_interaction(experiment_id, f"toggle_single_pieces_{display}", current_step)
    elif button_id == 'assembly-btn':
        display = 'block' if assembly_style.get('display') == 'none' else 'none'
        assembly_style = {'display': display}
        log_interaction(experiment_id, f"toggle_assembly_{display}", current_step)
    elif button_id == 'assembly-process-btn':
        display = 'block' if assembly_process_style.get('display') == 'none' else 'none'
        assembly_process_style = {'display': display}
        log_interaction(experiment_id, f"toggle_assembly_process_{display}", current_step)
    elif button_id == 'video-btn':
        display = 'block' if video_style.get('display') == 'none' else 'none'
        video_style = {'display': display}
        log_interaction(experiment_id, f"toggle_video_{display}", current_step)

    return (
        short_text_style, long_text_style, single_pieces_style,
        assembly_style, assembly_process_style, video_style
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