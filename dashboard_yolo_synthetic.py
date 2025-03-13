import pandas as pd
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from flask import send_file, jsonify
from openai import OpenAI
import os
from dash.exceptions import PreventUpdate
import dash_daq as daq
import dash
import cv2
import subprocess
from ultralytics import YOLO
import ast
import numpy as np
import rdflib, json
from scipy.stats import zscore

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

yolo_model = YOLO("yolo11n-pose.pt")

# Global variables
eeg_data = {ch: [] for ch in ["AF3", "AF4", "F3", "F4", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2", "C3", "C4"]}
hrv_data = []
timestamps = []
hrv_timestamps = []  # Independent timestamps for HRV
stimulus_active = False  # Flag to indicate active stimulus
stimulus_effect_duration = 2  # Duration of stimulus effect in seconds
stimuli_log = []  # To log stimulus type for each timestamp
sampling_window = 1000  # Number of data points to display (e.g., 1 second of HRV data)
total_recording_time = None  # Global variable to store total recording time

# Global variables
running = False
video_capture = None
video_writer = None
experiment_folder = ""
experiment_start_time = None
start_interval = None
end_interval = None



def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def create_dash_app(eeg_file_path, hrv_file_path):
    app = dash.Dash(__name__)


    global hrv_timestamps
    
    eeg_timestamps = eeg_data['Timestamp']
    eeg_start_time = eeg_timestamps.iloc[0]
    eeg_end_time = eeg_timestamps.iloc[-1]
    eeg_duration = eeg_end_time - eeg_start_time
    
    if 'Timestamp' in hrv_data.columns:
        hrv_data['Timestamp'] -= hrv_data['Timestamp'].min()
        hrv_timestamps = hrv_data['Timestamp'].tolist()
        #print(f"HRV Timestamps initialized: {hrv_timestamps[:5]}")  # Debugging info
    else:
        raise KeyError("HRV data is missing the 'Timestamp' column.")


    # Dropdown options for annotations
    ANNOTATION_OPTIONS = [
        {'label': 'Peak in EEG Activity', 'value': 'Peak in EEG Activity'},
        {'label': 'Dip in HRV', 'value': 'Dip in HRV'},
        {'label': 'High Variability in EEG', 'value': 'High Variability in EEG'},
        {'label': 'Correlation Between HRV and EEG', 'value': 'Correlation Between HRV and EEG'},
        {'label': 'Sudden Dip in EEG Activity', 'value': 'Sudden Dip in EEG Activity'},
        {'label': 'Peak in HRV with Concurrent EEG Increase', 'value': 'Peak in HRV with Concurrent EEG Increase'}
    ]

    # Calculate total recording time from the video
    global total_recording_time
    total_recording_time = eeg_duration
    #print(f"Calculated Total Recording Time: {total_recording_time} seconds")

    #print("EEG Data Head:", eeg_data.head())
    #print("HRV Data Head:", hrv_data.head())
    #print("Pose Data Head:", pose_data.head())

    # Dropdown Query Categories
    query_options = {
        "Cognitive Load & Stress": [
            "How does EEG activity correlate with cognitive stress?",
            "Does HRV indicate physiological stress responses?",
            "Compare cognitive load trends over time."
        ],
        "HRV & Physiological States": [
            "Are there significant HRV anomalies?",
            "How does HRV variability change with EEG states?",
            "Does stable HRV suggest a relaxed physiological state?"
        ],
        "Motion & Activity Patterns": [
            "How does body movement correlate with EEG stress markers?",
            "Is there a relationship between motion variability and HRV?",
            "What kind of movement pattern was detected?"
        ],
        "Anomaly Detection & Root Causes": [
            "What is the most significant anomaly in EEG, HRV, and Pose correlation, and what are its possible causes?"
            "Drill down on the most significant anomaly detected—what caused it?",
            "Are there correlations between detected anomalies in EEG, HRV, and Pose data?",
            "What time intervals had the most anomalies?"
        ]
    }

    # Flatten the dropdown options for Dash
    dropdown_options = [
        {"label": f"{category}: {query}", "value": query}
        for category, queries in query_options.items()
        for query in queries
    ]

    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='eeg-hrv-graph',
                style={'height': '1000px'},
                config={'modeBarButtonsToAdd': ['select2d', 'lasso2d']},
            )
        ], style={'flex': '2', 'padding': '10px', "background": "#EEE", "flex-wrap" : "wrap"}),

        html.Div([
            dcc.RangeSlider(
                id="video-seek-range-slider",
                min=0,
                max=total_recording_time if total_recording_time else 0,
                step=0.1,
                value=[0, total_recording_time if total_recording_time else 0],
                marks={
                    0: {"label": "0", "style": {"color": "black", "font-family": "Arial"}},
                    total_recording_time if total_recording_time else 0: {
                        "label": f"{total_recording_time:.1f}" if total_recording_time else "0",
                        "style": {"color": "black", "font-family": "Arial"}}
                },
                className="custom-slider",
            ),
            html.Div(
                id="range-slider-value-display",
                style={"textAlign": "center", "marginTop": "-10px", "font-family": "Arial", "font-weight": "bold",
                       'font-size': '13px'}
            ),
            html.Div([
                html.Div([
                daq.ColorPicker(
                    id='annotation-color-picker',
                    label='Select Annotation Color',
                    value={'hex': '#2E91E5'},  # Default color (blue)
                    style={'margin-bottom': '10px', "font-family": "Arial", "font-weight": "bold"}
                ),
                html.Div([
                html.P(
                    "Select your desired annotation range using the slider above, then enter your annotation below and click 'Add Annotation'",
                    style={"fontSize": "14px", "color": "#OOO", "marginBottom": "5px", "display": "block",
                           "width": "100%", "font-family": "Arial", "font-weight": "bold"}
                ),
                dcc.Dropdown(
                    id='annotation-dropdown',
                    options=ANNOTATION_OPTIONS,
                    placeholder='Select an annotation type',
                    style={'width': '100%', "font-family": "Arial", "font-weight": "bold", "font-size": "12px"}
                ),
                html.Button(
                    'Add Annotation',
                    id='add-annotation-button',
                    n_clicks=0,
                    style={
                        "padding": "10px 15px",
                        "borderRadius": "5px",
                        "border": "none",
                        "backgroundColor": "#673AB7",
                        "color": "white",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "transition": "background-color 0.3s",
                        "font-weight": "bold"
                    }),
                    html.P(
                        "Click 'Smart Annotations' to automatically analyze the data and suggest annotation ranges.",
                        style={"fontSize": "14px", "color": "#OOO", "marginBottom": "5px", "display": "block",
                               "width": "100%", "font-family": "Arial", "font-weight": "bold"}
                    ),
                    html.Button(
                        'NeuroSyncAI Automated Annotations',
                        id='smart-annotations-button',
                        n_clicks=0,
                        style={
                            "padding": "10px 15px",
                            "borderRadius": "5px",
                            "border": "none",
                            "backgroundColor": "#4CAF50",
                            "color": "white",
                            "cursor": "pointer",
                            "fontSize": "14px",
                            "transition": "background-color 0.3s",
                            "font-weight": "bold"
                        }
                    ),
                ], style= {
                        "width": "100%",
                        "display": "flex",
                        "flex-direction": "column",
                        "flex-wrap": "nowrap",
                        "align-content": "space-between",
                        "justify - content": "flex-start",
                        "row-gap": "10px",
                        "padding": "0px 10px"
                    }
                    )],
                    style={
                        "display" : "flex"
                    }),
                dcc.Loading(
                    id="loading-llm-output",
                    type="default",
                    children=[
                        html.Div(
                            id='llm-annotation-suggestions',
                            style={
                                "marginTop": "20px",
                                "font-size": "13px",
                                "line-height": "16px",
                                "padding": "10px",
                                "border": "1px solid #ccc",
                                "borderRadius": "0px",
                                "white-space": "pre-wrap",
                                "background-color": "#fff"
                            }
                        )
                    ],
                ),
                html.Button(
                    'Add to Plot',
                    id='add-to-plot-button',
                    n_clicks=0,
                    style={
                        "padding": "10px 15px",
                        "borderRadius": "5px",
                        "border": "none",
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "font-weight": "bold",
                        "marginTop": "10px",
                        "display": "none"
                    }
                ),
                html.P(
                    "Select the annotation you want to delete and click 'Delete Annotation'",
                    style={"fontSize": "14px", "color": "#OOO", "marginBottom": "5px", "display": "block",
                           "width": "100%", "font-family": "Arial", "font-weight": "bold"}
                ),
                dcc.Dropdown(
                    id='annotation-selector',
                    placeholder='Select annotation to delete',
                    options=[],  # Populated dynamically with annotations
                    style={'width': '100%'}
                ),
                html.Button('Delete Annotation', id='delete-annotation-button',
                        style={"padding": "10px 15px",
                        "borderRadius": "5px",
                        "border": "none",
                        "backgroundColor": "red",
                        "color": "white",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "font-weight": "bold",
                        "marginTop": "10px"}),
                dcc.Store(id='static-annotations-store'),
                dcc.Store(id='smart-annotations-store'),
                dcc.Store(id='annotations-store', data=[]),  # Store for annotations
                dcc.Store(id='selected-interval-store', data=[]),  # Store for selected interval
            ], style={
                "marginTop": "20px",
                "padding": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "backgroundColor": "#fff",
                "font-family": "Arial",
                "fontSize": "14px",
                "color": "#333",
                "row-gap": "10px",
                "display": "flex",
                "flex-direction": "column"
            }),

        ], style={'flex': '1', 'padding': '10px', "background": "#EEE"}),

        # Adding the Query Box for LLM integration
        html.Div([
            html.P(
                "Ask a question about the EEG, HRV or Pose data:",
                style={"fontSize": "14px", "color": "#000", "marginBottom": "5px", "display": "block",
                       "font-family": "Arial", "font-weight": "bold", "margin-bottom": "10px"}
            ),
            dcc.Dropdown(
                id="query-input",
                options=dropdown_options,
                placeholder="Select a question and click the submit button and NeuroSyncAI will answer!",
                style={
                    "borderRadius": "5px",
                    "border": "1px solid #ccc",
                    "fontSize": "14px",
                    "marginRight": "10px",
                    "float": "left",
                    'width': '700px',
                    "font-family": "Arial",
                }
            ),
            html.Button(
                'Submit Query',
                id='submit-query-button',
                n_clicks=0,
                style={
                    "padding": "10px 15px",
                    "borderRadius": "5px",
                    "border": "none",
                    "backgroundColor": "rgb(255 201 102)",
                    "color": "black",
                    "cursor": "pointer",
                    "fontSize": "14px",
                    "transition": "background-color 0.3s",
                    "font-weight": "bold"
                }
            ),
            dcc.Loading(
                id="loading-llm-output-data-query",
                show_initially="Select a question and click the submit button and NeuroSyncAI will answer!",
                type="default",
                children=[
                    html.Pre(
                        id='query-response',
                        style={
                            "marginTop": "20px",
                            "padding": "10px",
                            "border": "1px solid #ddd",
                            "borderRadius": "5px",
                            "backgroundColor": "#fff",
                            "font-family": "Arial",
                            "fontSize": "14px",
                            "color": "#333",
                            "min-height": "300px"
                        }
                    )]
            )
        ], style={
            "marginTop": "10px",
            "padding": "10px",
            "backgroundColor": "#EEE",
            "width" : "calc(100% - 20px)"
        }),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'width': '100%',
        'justifyContent': 'space-between',
        'flex-wrap': 'wrap',
        'marginTop': '20px'  # Adjust the value as needed
        })

    @app.callback(
        Output('selected-interval-store', 'data'),
        [Input('eeg-hrv-graph', 'selectedData'),
         Input('video-seek-range-slider', 'value')],
        [State('selected-interval-store', 'data')]
    )
    def update_selected_interval_store(selected_data, slider_value, current_interval):
        if selected_data and 'range' in selected_data:
            # Extract the range from the graph selection
            x_range = selected_data['range']['x']
            start_value, end_value = x_range[0], x_range[1]
            return {'start': start_value, 'end': end_value}

        elif slider_value:
            # If no graph selection, use slider values
            start_value, end_value = slider_value[0], slider_value[1]
            return {'start': start_value, 'end': end_value}

        # Default to the current interval if no input is provided
        return current_interval if current_interval else {'start': 0, 'end': 0}

    @app.callback(
        Output('range-slider-value-display', 'children'),
        [Input('selected-interval-store', 'data')]
    )
    def update_selected_interval_display(selected_interval):
        if selected_interval and 'start' in selected_interval and 'end' in selected_interval:
            start, end = selected_interval['start'], selected_interval['end']
            return f"Selected Interval: {start:.1f}s - {end:.1f}s"

        return "Selected Interval: 0.0s - 0.0s"

    def save_annotations_to_csv(annotations, eeg_file, hrv_file, pose_file):
        """
        Saves smart annotations into EEG, HRV, and Pose CSV files.
        """

        for annotation in annotations:
            start, end, text = annotation["start"], annotation["end"], annotation["full_text"]

            # Ensure "Annotation" column exists
            for df in [eeg_data, hrv_data, pose_data]:
                if "Annotation" not in df.columns:
                    df["Annotation"] = ""

            # Apply annotations within the relevant time range
            eeg_data.loc[(eeg_data["Timestamp"] >= start) & (eeg_data["Timestamp"] <= end), "Annotation"] = text
            hrv_data.loc[(hrv_data["Timestamp"] >= start) & (hrv_data["Timestamp"] <= end), "Annotation"] = text
            pose_data.loc[(pose_data["Timestamp"] >= start) & (pose_data["Timestamp"] <= end), "Annotation"] = text

            # Save back to CSV
            eeg_data.to_csv(eeg_file, index=False)
            hrv_data.to_csv(hrv_file, index=False)
            pose_data.to_csv(pose_file, index=False)

    @app.callback(
        Output('static-annotations-store', 'data'),
        [Input('add-annotation-button', 'n_clicks')],
        [State('selected-interval-store', 'data'),
         State('annotation-dropdown', 'value'),
         State('static-annotations-store', 'data'),
         State('annotation-color-picker', 'value')]
    )
    def add_static_annotation(n_clicks, selected_interval, annotation_value, current_annotations, color):
        if not n_clicks or not selected_interval or not annotation_value:
            raise PreventUpdate

        current_annotations = current_annotations or []
        start, end = selected_interval.get('start', 0), selected_interval.get('end', 0)

        if start != end:
            new_annotation = {'start': start, 'end': end, 'text': annotation_value, 'color': color['hex']}
            current_annotations.append(new_annotation)

        return current_annotations

    @app.callback(
        Output('annotations-store', 'data'),
        [
            Input('static-annotations-store', 'data'),
            Input('smart-annotations-store', 'data')
        ]
    )
    def update_annotations(static_annotations, smart_annotations):
        """
        Combines static annotations and smart annotations into a unified list.
        """
        # Safeguard against None inputs
        static_annotations = static_annotations or []
        smart_annotations = smart_annotations or []

        # Combine both annotations, ensuring no duplication
        combined_annotations = static_annotations + [
            annotation for annotation in smart_annotations
            if annotation not in static_annotations
        ]

        return combined_annotations

    def detailed_temporal_summary(eeg_data, hrv_data, pose_data, aggregation_window=5.0):
        """
        Summarizes EEG, HRV, and Pose data for the given interval to be used by the LLM.
        """
        summary = []

        # Ensure 'Timestamp' column is present
        for df, name in [(eeg_data, "EEG"), (hrv_data, "HRV")]:
            if 'Timestamp' not in df.columns:
                raise ValueError(f"Expected 'Timestamp' column in {name} data.")

        # Convert numeric columns to float and drop non-numeric values
        for data, label in [(eeg_data, "EEG"), (hrv_data, "HRV")]:
            numeric_columns = data.select_dtypes(include=['number']).columns.difference(['Timestamp'])
            data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
            data.dropna(inplace=True)
            
            for channel in numeric_columns:
                mean_val = data[channel].mean()
                std_val = data[channel].std()
                threshold_high = mean_val + 3 * std_val
                threshold_low = mean_val - 3 * std_val
                
                for idx, value in data[channel].items():
                    timestamp = data.loc[idx, 'Timestamp']
                    if value > threshold_high:
                        summary.append(f"{timestamp:.2f}s - Sudden peak in {label} channel {channel} detected")
                    elif value < threshold_low:
                        summary.append(f"{timestamp:.2f}s - Sudden dip in {label} channel {channel} detected")

        # --- Pose Data Analysis ---
        joint_names = ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", 
                    "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
        
        for idx, row in enumerate(pose_data):
            if not isinstance(row, dict):
                continue  # Skip malformed rows

            timestamp = row.get('Timestamp', None)
            keypoints = row.get('keypoints', [])
            
            if isinstance(keypoints, str):
                keypoints = ast.literal_eval(keypoints)  # Convert string to list safely
            
            if isinstance(keypoints, list) and len(keypoints) > 0 and isinstance(keypoints[0], list):
                keypoints = keypoints[0]  # Extract the first level of nested lists
            
            if not isinstance(keypoints, list) or len(keypoints) != len(joint_names):
                continue  # Skip malformed pose data
            
            for joint_idx, joint_name in enumerate(joint_names):
                x, y = keypoints[joint_idx]
                
                # Detect rapid movement or sudden stillness
                if idx > 0 and isinstance(pose_data[idx - 1], dict):
                    prev_x, prev_y = pose_data[idx - 1]['keypoints'][joint_idx]
                    movement = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
                    
                    if movement > 20:
                        summary.append(f"{timestamp:.2f}s - Drastic movement detected at {joint_name}")
                    elif movement < 5:
                        summary.append(f"{timestamp:.2f}s - Sudden stillness detected at {joint_name}")

                # Detect abnormal posture (e.g., extreme angles)
                if y < 50:
                    summary.append(f"{timestamp:.2f}s - Unusual posture detected at {joint_name}")
        
        return summary


    @app.callback(
        [Output('smart-annotations-store', 'data'),
         Output('llm-annotation-suggestions', 'children')],
        [Input('smart-annotations-button', 'n_clicks')],
        [State('annotations-store', 'data'),
         State('video-seek-range-slider', 'value'),
        State('annotation-color-picker', 'value')]
    )
    
    def smart_annotations(n_clicks, current_annotations, slider_value, color):
        if n_clicks == 0:
            return [], ""

        anomaly_summary = generate_anomaly_summary(eeg_data, hrv_data, pose_data)

        print("ANOMALY SUMMARY", anomaly_summary)

        prompt = f"""Given the provided EEG, HRV, and Pose data with corresponding timestamps, identify any notable anomalies or events. 
        For each notable event, provide a start and end time (in seconds) and a brief annotation of what was observed. 
        Keep the output concise and in the following format and don't include any introductory sentences or summaries in your response:

            [start time] - [end time]: [annotation]

            For example:
            0:50 - 1:05: Sudden HRV elevation detected
            1:45 - 2:00: Low theta activity in EEG channels F3, F7
        
        Focus on the following:
        - Sudden changes (peaks or dips) in HRV and EEG.
        - Anomalies in Pose keypoints, including rapid movement or unexpected stillness.
        - Correlations between HRV, EEG, and Pose data.
        - Ensure that Pose anomalies are given equal attention as EEG and HRV. Identify movements, postural changes, and correlations explicitly.
        
        Data Summary:
        {anomaly_summary}
        """

        response = call_llm_api(prompt)

        if isinstance(response, dict) and 'children' in response:
            response = response['children']

        if not isinstance(response, str):
            response = str(response)

        smart_annotations = []
        ANNOTATION_LABELS = {
            "Sudden peak in EEG": "Peak",
            "Sudden dip in EEG": "Dip",
            "HRV spike detected": "Spike",
            "HRV drop detected": "Drop",
            "Drastic movement detected": "Movement",
            "Unusual posture detected": "Posture"
        }

        # Define max length for displayed annotation
        MAX_ANNOTATION_LENGTH = 30  # Adjust as needed

        smart_annotations = []
        for line in response.split("\n"):
            if not line.strip():
                continue  # Skip empty lines

            try:
                times, annotation = line.split(": ", 1)
                start, end = map(str.strip, times.split("-"))
                start = float(start.replace(":", "."))
                end = float(end.replace(":", "."))

                # Truncate annotation text if too long
                short_text = annotation if len(annotation) <= MAX_ANNOTATION_LENGTH else annotation[:MAX_ANNOTATION_LENGTH] + "..."
                smart_annotations.append({
                    "start": start,
                    "end": end,
                    "text": short_text,  # Shortened version for plot
                    "full_text": annotation,  # Full text for LLM response
                    "color": color['hex']
                })

            except ValueError as e:
                print(f"Error parsing line: {line}, Error: {e}")  # Debugging


        display_text = "\n".join([f"{a['start']} - {a['end']}: {a['full_text']}" for a in smart_annotations])

        save_annotations_to_csv(smart_annotations, eeg_file_path, hrv_file_path, pose_file_path)

        return smart_annotations, display_text

    @app.callback(
        Output('annotation-selector', 'options'),
        [Input('annotations-store', 'data')]
    )
    def populate_annotation_selector(annotations_store):
        if not annotations_store:
            return []
        return [{'label': f"{annotation['text']} ({annotation['start']}s - {annotation['end']}s)",
                 'value': i} for i, annotation in enumerate(annotations_store)]

    @app.callback(
        Output('annotations-store', 'data', allow_duplicate=True),
        [Input('delete-annotation-button', 'n_clicks')],
        [State('annotation-selector', 'value'),
         State('annotations-store', 'data')],
        prevent_initial_call=True
    )
    def delete_annotation(n_clicks, selected_annotation, annotations_store):
        if not n_clicks or selected_annotation is None or not annotations_store:
            raise PreventUpdate

        # Remove the selected annotation
        annotations_store.pop(selected_annotation)
        print(f"Deleted Annotation Index: {selected_annotation}")
        return annotations_store
    
    # Load Ontology and Convert to JSON-LD
    def load_ontology(file_path):
        g = rdflib.Graph()
        g.parse(file_path, format="turtle")
        return g

    def convert_to_jsonld(ontology_graph):
        json_data = []
        for subj, pred, obj in ontology_graph:
            json_data.append({
                "subject": subj.split("/")[-1],
                "predicate": pred.split("/")[-1],
                "object": obj.split("/")[-1]
            })
        return json.dumps(json_data, indent=4)

    ontology_graph = load_ontology("dashboard\ontology.ttl")
    ontology_json = convert_to_jsonld(ontology_graph)

    def extract_pose_data(pose_data):
        """Processes the pose data into a usable format for visualization."""
        joint_names = [
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", 
            "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]
        timestamps = pose_data['Timestamp']
        joint_positions = {joint: pose_data[joint].tolist() for joint in joint_names}
        return timestamps, joint_positions

    @app.callback(
        Output('eeg-hrv-graph', 'figure'),
        [Input('annotations-store', 'data'),
        Input('video-seek-range-slider', 'value')],
        [State('annotation-dropdown', 'value'),
        State('selected-interval-store', 'data')]
    )
    def update_graph_with_pose(annotations_store, slider_value, annotation_text, selected_interval):

        # Load saved annotations from CSVs
        saved_annotations = []
        for df, source in [(eeg_data, "EEG"), (hrv_data, "HRV"), (pose_data, "Pose")]:
            if "Annotation" in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row["Annotation"]) and row["Annotation"].strip():
                        saved_annotations.append({
                            "start": row.get("Timestamp", row.get("Timestamp", 0)),
                            "end": row.get("Timestamp", row.get("Timestamp", 0)) + 1,  # Approximate range
                            "text": row["Annotation"],
                            "color": "#2E91E5"  # Default color, could be improved by saving color info
                        })

        # Filter data based on slider range
        start_time, end_time = slider_value
        eeg_filtered = eeg_data[(eeg_data['Timestamp'] >= start_time) & (eeg_data['Timestamp'] <= end_time)]
        hrv_filtered = hrv_data[(hrv_data['Timestamp'] >= start_time) & (hrv_data['Timestamp'] <= end_time)]
        pose_filtered = pose_data[(pose_data["Timestamp"] >= start_time) & (pose_data["Timestamp"] <= end_time)]

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("EEG Data", "HRV Data", "Pose Keypoints")
        )

        # Add EEG data
        for column in eeg_filtered.columns[1:]:
            fig.add_trace(
                go.Scatter(
                    x=eeg_filtered['Timestamp'],
                    y=eeg_filtered[column],
                    mode='lines',
                    name=f"EEG: {column}"
                ),
                row=1, col=1
            )

        # Add HRV data
        fig.add_trace(
            go.Scatter(
                x=hrv_filtered['Timestamp'],
                y=hrv_filtered['HRV'],  # Replace 'HRV' with actual HRV column if needed
                mode='lines',
                name="HRV"
            ),
            row=2, col=1
        )

        # Add Pose Keypoints data
        joint_names = [
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]
        for joint_name in joint_names:
            if joint_name in pose_filtered.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pose_filtered['Timestamp'],
                        y=pose_filtered[joint_name],
                        mode='lines+markers',
                        name=f"Pose: {joint_name}"
                    ),
                    row=3, col=1
                )

        # Add Annotations
        if annotations_store:
            saved_annotations.extend(annotations_store)
            for annotation in annotations_store:
                start = annotation['start']
                end = annotation['end']
                text = annotation['text']
                color = annotation['color']

                # Calculate y0 and y1 dynamically for each subplot
                y_ranges = {
                    1: (
                        eeg_filtered.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').min().min(),
                        eeg_filtered.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').max().max()
                    ),  # EEG
                    2: (
                        pd.to_numeric(hrv_filtered['HRV'], errors='coerce').min(),
                        pd.to_numeric(hrv_filtered['HRV'], errors='coerce').max()
                    ),  # HRV
                    3: (
                        0,
                        350
                    )  # Pose Keypoints
                }

                #print("POSE FILTETERED", pose_filtered['keypoints'])

                for row_idx in range(1, 4):  # Rows 1 (EEG), 2 (HRV), and 3 (Pose)
                    y_min, y_max = y_ranges[row_idx]  # Get y-axis range for this row
                    fig.add_shape(
                        type="rect",
                        x0=start,
                        x1=end,
                        y0=y_min,  # Dynamic minimum
                        y1=y_max,  # Dynamic maximum
                        fillcolor=color,  # Use selected annotation color
                        opacity=0.4,  # Adjust opacity for better visibility
                        line=dict(width=0),  # No border for the rectangle
                        xref="x",  # Shared x-axis
                        yref=f"y{row_idx}"  # Reference to the respective subplot's y-axis
                    )
                    # Optional: Add annotation text at the center of the rectangle
                    fig.add_annotation(
                        x=(start + end) / 2,
                        y=y_max,  # Position annotation text at the top of the range
                        text=text,
                        showarrow=False,
                        font=dict(size=10, color=color),
                        xref="x",
                        yref=f"y{row_idx}"
                    )

        fig.update_layout(
            title="EEG, HRV, and Pose Keypoints with Stimuli and Annotations",
            yaxis_title="Signal Value",
            template="plotly_white",
             margin=dict(l=20, r=20, t=70, b=50),  # Adjust left (l), right (r), top (t), and bottom (b) margins
           legend=dict(
                orientation="h",  # Make the legend horizontal
                x=0.5,            # Center the legend horizontally
                y=-0.05,           # Position the legend below the chart
                xanchor="center", # Anchor the center of the legend
                yanchor="top"     # Align the top of the legend
            )
        )

        return fig

    # EEG Frequency Bands & Associated Cognitive States
    EEG_BANDS = {
        "Delta (0.5-4Hz)": "Deep Sleep / Unconscious",
        "Theta (4-8Hz)": "Meditation / Drowsiness",
        "Alpha (8-12Hz)": "Relaxation / Calm Focus",
        "Beta (12-30Hz)": "Active Thinking / Stress",
        "Gamma (30+Hz)": "High-Level Cognitive Processing"
    }

    # Motion Classification Thresholds
    MOTION_LOW_STD = 20  # Still posture
    MOTION_HIGH_STD = 100  # High movement

    ### EEG, HRV, and POSE ANOMALY DETECTION ###
    def detect_eeg_anomalies(eeg_df):
        """
        Identifies sudden peaks, dips, or variability in EEG signals.
        Returns a structured list of detected anomalies with timestamps.
        """
        anomalies = []

        if 'Timestamp' not in eeg_df.columns:
            raise ValueError("Expected 'Timestamp' column in EEG data.")

        numeric_columns = eeg_df.select_dtypes(include=['number']).columns.difference(['Timestamp'])

        for channel in numeric_columns:
            mean_val = eeg_df[channel].mean()
            std_val = eeg_df[channel].std()
            threshold_high = mean_val + 3 * std_val  # Upper threshold (outliers)
            threshold_low = mean_val - 3 * std_val  # Lower threshold (outliers)

            for idx, value in eeg_df[channel].items():
                timestamp = eeg_df.loc[idx, 'Timestamp']
                if value > threshold_high:
                    anomalies.append((f"EEG {channel} - Peak Detected", timestamp, timestamp))
                elif value < threshold_low:
                    anomalies.append((f"EEG {channel} - Dip Detected", timestamp, timestamp))

        return anomalies if anomalies else [("No EEG Anomalies Detected", "N/A", "N/A")]

    def detect_hrv_anomalies(hrv_df):
        """
        Identifies anomalies in HRV data, including sudden drops, instability, and extreme values.
        Returns a structured list of detected anomalies with timestamps.
        """
        anomalies = []

        if 'Timestamp' not in hrv_df.columns:
            raise ValueError("Expected 'Timestamp' column in HRV data.")

        numeric_columns = hrv_df.select_dtypes(include=['number']).columns.difference(['Timestamp'])

        for channel in numeric_columns:
            mean_val = hrv_df[channel].mean()
            std_val = hrv_df[channel].std()
            threshold_high = mean_val + 2.5 * std_val  # Relaxation (parasympathetic dominance)
            threshold_low = mean_val - 2.5 * std_val  # Stress (sympathetic dominance)

            for idx, value in hrv_df[channel].items():
                timestamp = hrv_df.loc[idx, 'Timestamp']

                if value > threshold_high:
                    anomalies.append((f"HRV {channel} - Unusually High HRV (Relaxation)", timestamp, timestamp))
                elif value < threshold_low:
                    anomalies.append((f"HRV {channel} - Unusually Low HRV (Stress)", timestamp, timestamp))

                # Additional logic for instability (fluctuations)
                if idx > 0:
                    prev_value = hrv_df.loc[idx - 1, channel]
                    change = abs(value - prev_value)
                    if change > std_val * 2:
                        anomalies.append((f"HRV {channel} - Sudden Fluctuation Detected", timestamp, timestamp))

        return anomalies if anomalies else [("No HRV Anomalies Detected", "N/A", "N/A")]

    def detect_pose_anomalies_windowed(pose_data, window_size=3):
        """
        Detects gait and movement anomalies in a windowed approach (default 3s windows).
        Focuses on stride variability, arm asymmetry, instability, and phase shifts.
        """

        if 'Timestamp' not in pose_data.columns:
            return [("Invalid Data: No Timestamp Found", "N/A", "N/A")]

        pose_data = pose_data.fillna(0)  # Handle missing values
        timestamps = pose_data["Timestamp"].values

        # Convert timestamps into seconds and segment into windows
        pose_data["Window"] = (pose_data["Timestamp"] // window_size).astype(int)

        anomalies = []

        # Key joints for analysis
        key_joints = {
            "stride": ["Left Hip", "Right Hip"],
            "arm_swing": ["Left Shoulder", "Right Shoulder"],
            "instability": ["Left Ankle", "Right Ankle"],
            "step_phase": ["Left Knee", "Right Knee"],
        }

        # Process each window separately
        grouped = pose_data.groupby("Window")
        for window, group in grouped:
            t_start, t_end = group["Timestamp"].iloc[0], group["Timestamp"].iloc[-1]

            # 1️⃣ **Stride Length Variability**
            if all(j in group.columns for j in key_joints["stride"]):
                stride_diff = np.abs(group["Left Hip"] - group["Right Hip"])
                if stride_diff.mean() < 10:  # Shortened stride
                    anomalies.append((f"{t_start}-{t_end}s - Reduced stride length detected", t_start, t_end))

            # 2️⃣ **Arm Swing Asymmetry**
            if all(j in group.columns for j in key_joints["arm_swing"]):
                arm_diff = np.abs(group["Left Shoulder"] - group["Right Shoulder"])
                if arm_diff.mean() > 15:  # Uneven arm swings
                    anomalies.append((f"{t_start}-{t_end}s - Arm swing asymmetry detected", t_start, t_end))

            # 3️⃣ **Instability Detection (Jitter in Ankles)**
            if all(j in group.columns for j in key_joints["instability"]):
                ankle_movement = (group["Left Ankle"] + group["Right Ankle"]) / 2
                instability_score = np.abs(np.diff(ankle_movement)).mean()
                if instability_score > 5:  # Detect erratic movement
                    anomalies.append((f"{t_start}-{t_end}s - Instability detected", t_start, t_end))

            # 4️⃣ **Step Phase Shift Detection**
            if all(j in group.columns for j in key_joints["step_phase"]):
                knee_phase_diff = np.abs(group["Left Knee"] - group["Right Knee"])
                if knee_phase_diff.mean() > np.pi/6:  # Delayed stepping phase shift
                    anomalies.append((f"{t_start}-{t_end}s - Step phase shift detected", t_start, t_end))

        return anomalies if anomalies else [("No Significant Pose Anomalies Detected", "N/A", "N/A")]


    ### STRUCTURED DATA PREPARATION ###
    def generate_anomaly_summary(eeg_data, hrv_data, pose_data):
        eeg_anomalies = detect_eeg_anomalies(eeg_data) if not eeg_data.empty else []
        hrv_anomalies = detect_hrv_anomalies(hrv_data) if not hrv_data.empty else []
        pose_anomalies = detect_pose_anomalies_windowed(pose_data, 3) if not pose_data.empty else []

        structured_summary = {
            "EEG": {"EEG Anomalies": eeg_anomalies},
            "HRV": {"HRV Anomalies": hrv_anomalies},
            "Pose": {"Pose Anomalies": pose_anomalies}
        }
        return json.dumps(structured_summary, indent=4)

    def detect_anomalies(data_series, threshold=3):
        return list(data_series.index[np.abs(zscore(data_series)) > threshold])

    def extract_pose_variability(pose_filtered):
        """
        Extracts the standard deviation of keypoint movements (pose variability).
        """
        joint_names = [
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]

        keypoint_variations = {}

        for joint_idx, joint_name in enumerate(joint_names):
            x_values, y_values = [], []

            for _, row in pose_filtered.iterrows():
                keypoints = row.get('keypoints', [])
                if isinstance(keypoints, str):
                    keypoints = ast.literal_eval(keypoints)  # Convert string to list
                
                if isinstance(keypoints, list) and len(keypoints) > joint_idx:
                    keypoint = keypoints[joint_idx]  # Extract joint position
                    if isinstance(keypoint, (list, tuple)) and len(keypoint) >= 2:
                        x, y = keypoint[:2]  # Extract x, y coordinates
                        x_values.append(x)
                        y_values.append(y)

            if x_values and y_values:
                keypoint_variations[joint_name] = {
                    "std_x": np.std(x_values),
                    "std_y": np.std(y_values)
                }

        return keypoint_variations

    @app.callback(
        Output('query-response', 'children'),
        [Input('submit-query-button', 'n_clicks')],
        [State('query-input', 'value'),
        State('annotations-store', 'data'),
        State('video-seek-range-slider', 'value')]
    )
    def process_query(n_clicks, query, annotations, slider_value):
        if n_clicks == 0 or not query:
            raise PreventUpdate

        """
        Processes data, generates insights, and compares LLM outputs.
        """
        anomaly_summary = json.dumps({
            "EEG": detect_eeg_anomalies(eeg_data),
            "HRV": detect_hrv_anomalies(hrv_data),
            "Pose": detect_pose_anomalies_windowed(pose_data,3)
        }, indent=4)

        system_prompt = f"""
        You are an AI analyzing EEG, HRV, and Pose data.

        **Ontology Information**:
        {ontology_json}

        **Data Summary**:
        {anomaly_summary}

        **User Query**:
        {query}
        """

        response = call_llm_api(system_prompt)

        # return html.Pre(f"### LLM Response:\n{response}\n\n### Ontology JSON:\n{ontology_json}\n\n### Data Summary:\n{structured_summary}", style={"white-space": "pre-wrap", "font-size": "13px", "background-color": "#fff"})
        return html.Pre(f"{anomaly_summary},{response}", style={"white-space": "pre-wrap", "font-size": "13px", "background-color": "#fff"})

    # Function to call the OpenAI API
    def call_llm_api(query):
        try:
            # The system message will define the context for the AI (e.g., it's analyzing EEG and HRV data)
            system_message = {"role": "system",
                              "content": "You analyze EEG, HRV, and Pose data."}

            # The user message will contain the user's question/query
            user_message = {"role": "user", "content": query}

            # Make the API call using the new chat completion approach
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[system_message, user_message],
                max_tokens=700,
                temperature=0.7
            )

            # Extract the assistant's response
            return response.choices[0].message.content

        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    return app

if __name__ == "__main__":
    eeg_file_path = "data\eeg_data_Motor Task_MCI_20250307_221718.csv"
    hrv_file_path = "data\hrv_data_Motor Task_MCI_20250307_221718.csv"
    pose_file_path = "data\walking_pose_Motor Task_MCI_trial1_20250307_221718_590611.csv"

    eeg_data = pd.read_csv(eeg_file_path)
    hrv_data = pd.read_csv(hrv_file_path)
    pose_data = pd.read_csv(pose_file_path)

    #print(eeg_data.head)
    #print(hrv_data.head)

    app = create_dash_app(eeg_file_path, hrv_file_path)
    app.run_server(port=8052, use_reloader=True)
