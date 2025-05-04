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
                "Click below to analyze EEG, HRV, and Pose data and receive an AI-generated diagnosis:",
                style={"fontSize": "14px", "color": "#000", "marginBottom": "5px", "display": "block",
                       "font-family": "Arial", "font-weight": "bold", "margin-bottom": "10px"}
            ),
            # dcc.Dropdown(
            #     id="query-input",
            #     options=dropdown_options,
            #     placeholder="Select a question and click the submit button and NeuroSyncAI will answer!",
            #     style={
            #         "borderRadius": "5px",
            #         "border": "1px solid #ccc",
            #         "fontSize": "14px",
            #         "marginRight": "10px",
            #         "float": "left",
            #         'width': '700px',
            #         "font-family": "Arial",
            #     }
            # ),
            html.Button(
                'NeuroSyncAI Diagnosis Help',
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

    def eeg_band_anomaly_detector(eeg_df, sampling_rate=256, window_sec=2, alpha_thresh=0.2, beta_thresh=0.15, z_thresh=-2.0):
        from scipy.signal import welch
        from scipy.stats import zscore
        import numpy as np

        if 'Timestamp' not in eeg_df.columns:
            raise ValueError("Expected 'Timestamp' column in EEG data.")

        eeg_channels = eeg_df.select_dtypes(include=['number']).columns.difference(['Timestamp'])
        window_size = int(sampling_rate * window_sec)
        total_samples = len(eeg_df)
        num_windows = total_samples // window_size

        band_powers = {ch: {'theta': [], 'alpha': [], 'beta': []} for ch in eeg_channels}
        window_times = []

        for w in range(num_windows):
            start = w * window_size
            end = start + window_size
            seg = eeg_df.iloc[start:end]
            if len(seg) < window_size:
                continue

            window_times.append(seg['Timestamp'].mean())
            for ch in eeg_channels:
                freqs, psd = welch(seg[ch], fs=sampling_rate, nperseg=window_size)
                theta = np.trapezoid(psd[(freqs >= 4) & (freqs < 8)], freqs[(freqs >= 4) & (freqs < 8)])
                alpha = np.trapezoid(psd[(freqs >= 8) & (freqs < 12)], freqs[(freqs >= 8) & (freqs < 12)])
                beta = np.trapezoid(psd[(freqs >= 13) & (freqs < 30)], freqs[(freqs >= 13) & (freqs < 30)])

                band_powers[ch]['theta'].append(theta)
                band_powers[ch]['alpha'].append(alpha)
                band_powers[ch]['beta'].append(beta)

        anomalies = []

        # 1️⃣ Local band ratio suppression
        alpha_ratios, beta_ratios = [], []
        for ch in eeg_channels:
            for i, t in enumerate(window_times):
                th = band_powers[ch]['theta'][i]
                al = band_powers[ch]['alpha'][i]
                be = band_powers[ch]['beta'][i]
                total = th + al + be
                if total == 0:
                    continue
                alpha_ratio = al / total
                beta_ratio = be / total
                alpha_ratios.append(alpha_ratio)
                beta_ratios.append(beta_ratio)

                if alpha_ratio < alpha_thresh:
                    anomalies.append(f"Low alpha ratio in {ch} at {t:.2f}s (ratio = {alpha_ratio:.2f})")
                if beta_ratio < beta_thresh:
                    anomalies.append(f"Low beta ratio in {ch} at {t:.2f}s (ratio = {beta_ratio:.2f})")

        # Debug output: band ratio distributions
        print("Alpha ratio percentiles:", np.percentile(alpha_ratios, [5, 25, 50, 75, 95]))
        print("Beta ratio percentiles :", np.percentile(beta_ratios, [5, 25, 50, 75, 95]))

        # 2️⃣ Global z-score check
        for band in ['alpha', 'beta']:
            means = [np.mean(band_powers[ch][band]) for ch in eeg_channels]
            zs = zscore(means)
            for ch, z in zip(eeg_channels, zs):
                if z < z_thresh:
                    label = "Suppressed Alpha" if band == "alpha" else "Suppressed Beta"
                    anomalies.append(f"{label} globally in {ch} (z = {z:.2f})")

        return anomalies or ["No significant EEG band anomalies detected."]

    def detect_hrv_suppression(hrv_df, suppression_threshold=30):
        """
        Detects HRV suppression based on SDNN (standard deviation of NN intervals).
        Expects an 'HRV' column containing RR intervals in ms.
        """
        if 'HRV' not in hrv_df.columns:
            raise ValueError("Expected 'HRV' column with RR intervals in milliseconds.")

        sdnn = np.std(hrv_df['HRV'])

        print(f"\n[HRV] SDNN = {sdnn:.2f} ms — {'Suppressed' if sdnn < suppression_threshold else 'Normal'}")

        if sdnn < suppression_threshold:
            return f"HRV suppression detected (SDNN = {sdnn:.2f} ms, below {suppression_threshold} ms threshold)."
        else:
            return f"No significant HRV suppression detected (SDNN = {sdnn:.2f} ms)."

    def generate_position_based_pose_summary(pose_file):
        from scipy.signal import find_peaks
        """
        Updated summary generator for your current dataset using left-side joints.
        """
        try:
            df = pd.read_csv(pose_file)
            filename = os.path.basename(pose_file)
            label = "MCI" if "MCI" in filename else "Healthy"

            # Use LEFT joints for motion analysis
            joint_map = {
                "hip": "Left Hip",
                "knee": "Left Knee",
                "ankle": "Left Ankle"
            }

            dt = 1 / 30  # 30 FPS
            features = {
                "subject_id": filename.replace(".csv", ""),
                "label": label
            }

            for key, col in joint_map.items():
                if col not in df.columns:
                    print(f"⚠️ Missing joint: {col}")
                    features[f"{key}_mean"] = 0
                    features[f"{key}_std"] = 0
                    features[f"{key}_range"] = 0
                    features[f"{key}_acc_std"] = 0
                    features[f"{key}_jerk"] = 0
                    continue

                x = df[col].values
                velocity = np.gradient(x, dt)
                acceleration = np.gradient(velocity, dt)
                jerk = np.gradient(acceleration, dt)

                features[f"{key}_mean"] = np.mean(x)
                features[f"{key}_std"] = np.std(x)
                features[f"{key}_range"] = np.ptp(x)
                features[f"{key}_acc_std"] = np.std(acceleration)
                features[f"{key}_jerk"] = np.mean(np.abs(jerk))

            # Add cadence if both ankles and timestamps are available
            if all(col in df.columns for col in ["Left Ankle", "Right Ankle", "Timestamp"]):
                left = df["Left Ankle"].values
                right = df["Right Ankle"].values
                timestamps = df["Timestamp"].values

                if timestamps[-1] > 1000:
                    timestamps = timestamps / 1000

                duration = timestamps[-1] - timestamps[0]
                step_length = np.abs(np.diff(left) - np.diff(right))
                stride_length = np.abs(np.diff(left[::2]))
                peaks, _ = find_peaks(-left, distance=20)
                cadence = (len(peaks) / duration) * 60 if duration > 0 else 0

                features["step_length_mean"] = np.mean(step_length)
                features["stride_length_mean"] = np.mean(stride_length)
                features["cadence"] = cadence

            return features

        except Exception as e:
            print(f"❌ Error in generate_position_based_pose_summary for {pose_file}: {e}")
            return {"subject_id": os.path.basename(pose_file).replace(".csv", ""), "label": "Unknown", "error": str(e)}

    def generate_combined_summary(eeg_df, hrv_df, pose_summary):
        from scipy.signal import welch
        from scipy.stats import zscore
        import numpy as np

        # --- EEG SUMMARY ---
        eeg_channels = eeg_df.select_dtypes(include=['number']).columns.difference(['Timestamp'])
        window_size = 512
        band_powers = {ch: {'theta': [], 'alpha': [], 'beta': []} for ch in eeg_channels}
        for w in range(len(eeg_df) // window_size):
            seg = eeg_df.iloc[w * window_size:(w + 1) * window_size]
            for ch in eeg_channels:
                freqs, psd = welch(seg[ch], fs=256, nperseg=window_size)
                for band, lo, hi in [('theta', 4, 8), ('alpha', 8, 12), ('beta', 13, 30)]:
                    power = np.trapezoid(psd[(freqs >= lo) & (freqs < hi)], freqs[(freqs >= lo) & (freqs < hi)])
                    band_powers[ch][band].append(power)

        means = {band: np.mean([np.mean(band_powers[ch][band]) for ch in eeg_channels])
                for band in ['theta', 'alpha', 'beta']}
        ta_ratio = means['theta'] / (means['alpha'] + 1e-6)
        tb_ratio = means['theta'] / (means['beta'] + 1e-6)

        eeg_summary = []
        if means['alpha'] < 1.0:
            eeg_summary.append("Alpha power is globally reduced.")
        else:
            eeg_summary.append("Alpha power is within healthy range.")
        if means['theta'] > 0.3:
            eeg_summary.append("Theta activity is elevated.")
        else:
            eeg_summary.append("Theta levels are normal.")
        eeg_summary.append(f"Theta/Alpha ratio: {ta_ratio:.2f}")
        eeg_summary.append(f"Theta/Beta ratio: {tb_ratio:.2f}")

        # --- HRV SUMMARY ---
        if 'HRV' in hrv_df.columns:
            sdnn = np.std(hrv_df['HRV'])
            if sdnn < 30:
                hrv_summary = f"HRV suppression detected (SDNN = {sdnn:.2f} ms)."
            else:
                hrv_summary = f"HRV appears normal (SDNN = {sdnn:.2f} ms)."
        else:
            hrv_summary = "HRV data missing."

        # --- Pose SUMMARY ---
        pose_lines = []
        for k, v in pose_summary.items():
            if isinstance(v, float):
                pose_lines.append(f"{k.replace('_', ' ')} = {v:.2f}")
        pose_str = "; ".join(pose_lines)

        # --- Final Combined Summary ---
        return (
            "EEG Summary:\n" + "\n".join(eeg_summary) + "\n\n"
            "HRV Summary:\n" + hrv_summary + "\n\n"
            "Pose Summary:\n" + pose_str
        )

    def build_llm_prompt_from_combined_summary(combined_summary_text, user_query=None):
        prompt_parts = []

        # 🧠 MCI-Centric Unified Ontology
        prompt_parts.append("""
    Multimodal Ontology Reference:
    Mild Cognitive Impairment (MCI) may manifest through a combination of cognitive and subtle motor irregularities. Interpret the subject's condition based on the following:

    - EEG markers:
    - Increased Theta power and decreased Alpha power, especially in frontal and parietal regions, are commonly associated with MCI.
    - Suppressed Beta activity may indicate cognitive slowing.

    - HRV markers:
    - Reduced heart rate variability (e.g., SDNN < 30 ms) is linked to decreased autonomic regulation and cognitive decline.

    - Gait (pose-derived) markers:
    - Healthy gait patterns exhibit wide joint movement ranges, smooth transitions, and consistent acceleration.
    - MCI-related gait changes may include reduced range of motion, higher jerk (abrupt changes in motion), and greater acceleration variability across joints, even without overt motor disability.

    Diagnostic Guidance:
    - Classify as **MCI** if EEG and/or HRV features indicate cognitive dysfunction, especially when supported by mild motor irregularities.
    - Classify as **Control** if all modalities are within normal patterns.
    - Classify as **Needs clinical review** if findings are weak, borderline, or inconsistent.

    Use all three modalities in combination to support a holistic evaluation.
    """)

        # 🧪 Combined Summary
        prompt_parts.append("Multimodal Summary:\n" + combined_summary_text.strip())

        # 🧾 Task logic
        if user_query:
            prompt_parts.append(f"\nUser Question:\n{user_query.strip()}")
        else:
            prompt_parts.append("""
    Task:
    Based on the EEG, HRV, and Gait summaries provided above, assess the subject's cognitive-motor status.

    Respond using the following format:
    Conclusion: [MCI / Control / Needs clinical review]
    Explanation: [Concise reasoning using any or all modalities]
    """)

        return "\n\n".join(prompt_parts)
    
    @app.callback(
        Output('query-response', 'children'),
        [Input('submit-query-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def process_query(n_clicks):
        if n_clicks == 0:
            raise PreventUpdate

        try:
            # Step 1: Generate multimodal summary
            pose_summary = generate_position_based_pose_summary(pose_file_path)
            combined_summary = generate_combined_summary(eeg_data, hrv_data, pose_summary)

            # Step 2: Use LLM prompt with data + user query
            system_prompt = build_llm_prompt_from_combined_summary(combined_summary)
            full_prompt = f"{system_prompt}"

            # Step 3: Call LLM
            response = call_llm_api(full_prompt)

            return html.Pre(response, style={
                "white-space": "pre-wrap",
                "font-size": "13px",
                "background-color": "#fff",
                "padding": "10px"
            })

        except Exception as e:
            return html.Pre(f"❌ Error: {str(e)}", style={"white-space": "pre-wrap", "color": "red"})

    def call_llm_api(prompt_text):
        try:
            messages = [
                {"role": "system", "content": "You are an expert system interpreting EEG, HRV, and Pose summaries for neurocognitive diagnosis."},
                {"role": "user", "content": prompt_text}
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=700,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    return app

if __name__ == "__main__":
    eeg_file_path = "data\eeg_data_Baseline Resting_Healthy_trial16_20250503_155735_948601.csv"
    hrv_file_path = "data\hrv_data_Baseline Resting_Healthy_trial16_20250503_155740_097657.csv"
    pose_file_path = "data\walking_pose_Baseline Resting_Healthy_trial16_20250503_155740_162179.csv"

    eeg_data = pd.read_csv(eeg_file_path)
    hrv_data = pd.read_csv(hrv_file_path)
    pose_data = pd.read_csv(pose_file_path)

    #print(eeg_data.head)
    #print(hrv_data.head)

    app = create_dash_app(eeg_file_path, hrv_file_path)
    app.run_server(port=8052, use_reloader=True)
