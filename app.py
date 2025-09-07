import gradio as gr
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained model
try:
    with open('flight_price_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    # If model doesn't exist, train it
    print("Training model...")
    
    # Load and preprocess data
    data = pd.read_csv('flights.csv')
    
    # Fill missing values
    data['Route'] = data['Route'].fillna(data['Route'].mode()[0])
    data['Total_Stops'] = data['Total_Stops'].fillna(data['Total_Stops'].mode()[0])
    
    # Feature Engineering
    data['Date'] = data['Date_of_Journey'].str.split('/').str[0].astype(int)
    data['Month'] = data['Date_of_Journey'].str.split('/').str[1].astype(int)
    data['Year'] = data['Date_of_Journey'].str.split('/').str[2].astype(int)
    data.drop('Date_of_Journey', axis=1, inplace=True)
    
    def convert_duration(duration):
        if len(duration.split()) == 2:
            hours = int(duration.split()[0][:-1])
            minutes = int(duration.split()[1][:-1])
            return hours * 60 + minutes
        else:
            if 'h' in duration:
                return int(duration[:-1]) * 60
            else:
                return int(duration[:-1])
    
    data['Duration'] = data['Duration'].apply(convert_duration)
    
    # Label Encoding
    le_airline = LabelEncoder()
    le_source = LabelEncoder()
    le_destination = LabelEncoder()
    le_stops = LabelEncoder()
    
    data['Airline'] = le_airline.fit_transform(data['Airline'])
    data['Source'] = le_source.fit_transform(data['Source'])
    data['Destination'] = le_destination.fit_transform(data['Destination'])
    data['Total_Stops'] = le_stops.fit_transform(data['Total_Stops'])
    
    # Drop unnecessary columns
    data.drop(['Route', 'Additional_Info', 'Dep_Time', 'Arrival_Time'], axis=1, inplace=True)
    
    # Define features and target
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    with open('flight_price_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved!")

# Load original data to get unique values for dropdowns
data = pd.read_csv('flights.csv')

# Get unique values for dropdowns
airlines = sorted(data['Airline'].unique())
sources = sorted(data['Source'].unique())
destinations = sorted(data['Destination'].unique())
stops = sorted(data['Total_Stops'].dropna().unique())

# Create label encoders for prediction
le_airline = LabelEncoder()
le_source = LabelEncoder()
le_destination = LabelEncoder()
le_stops = LabelEncoder()

le_airline.fit(data['Airline'])
le_source.fit(data['Source'])
le_destination.fit(data['Destination'])
le_stops.fit(data['Total_Stops'])

import datetime

def predict_flight_price(airline, source, destination, duration_hours, duration_minutes, total_stops, date, month, year):
    # Date validation
    try:
        journey_date = datetime.date(year, month, date)
        if journey_date < datetime.date.today():
            return "Error: Cannot predict for past dates. Please select a future date."
    except ValueError:
        return "Error: Invalid date. Please enter a valid date."

    # Source and Destination validation
    if source == destination:
        return "Error: Source and Destination cities cannot be the same."

    """
    Predict flight price based on input parameters
    """
    try:
        # Convert duration to minutes
        duration = duration_hours * 60 + duration_minutes
        
        # Encode categorical variables
        airline_encoded = le_airline.transform([airline])[0]
        source_encoded = le_source.transform([source])[0]
        destination_encoded = le_destination.transform([destination])[0]
        stops_encoded = le_stops.transform([total_stops])[0]
        
        # Create feature array
        features = np.array([[airline_encoded, source_encoded, destination_encoded, 
                            duration, stops_encoded, date, month, year]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return f"₹{prediction:.2f}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Flight Price Predictor", css="""
    .gradio-container {
        background-color: #000000 !important;
        font-family: 'Arial', sans-serif !important;
    }
    .gr-button {
        background-color: #ff7f0e !important;
        border-color: #ff7f0e !important;
        color: white !important;
    }
    .gr-button:hover {
        background-color: #e67300 !important;
        border-color: #cc6600 !important;
    }
    .gr-form {
        background-color: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
    }
    .gr-textbox, .gr-dropdown {
        border: 1px solid #444 !important;
        border-radius: 4px !important;
        background-color: #222 !important;
        color: white !important;
    }
    h1 {
        color: #ff7f0e !important;
        text-align: center !important;
    }
    .gr-markdown {
        color: #cccccc !important;
    }
""") as demo:
    def update_destination_choices(source_city):
        available_destinations = [d for d in destinations if d != source_city]
        return gr.Dropdown(choices=available_destinations, value=available_destinations[0])
    gr.Markdown("# ✈️ Flight Price Predictor")
    gr.Markdown("Predict flight prices based on various parameters using machine learning.")
    
    with gr.Row():
        with gr.Column():
            airline = gr.Dropdown(
                choices=airlines,
                label="Airline",
                value=airlines[0]
            )
            
            source = gr.Dropdown(
                choices=sources,
                label="Source City",
                value=sources[0]
            )
            
            destination = gr.Dropdown(
                choices=destinations,
                label="Destination City",
                value=destinations[0]
            )
            
            total_stops = gr.Dropdown(
                choices=stops,
                label="Total Stops",
                value=stops[0]
            )
    
    source.change(update_destination_choices, inputs=source, outputs=destination)
    
    with gr.Column():
        with gr.Row():
            duration_hours = gr.Number(
                label="Flight Duration (Hours)",
                value=2,
                minimum=0,
                maximum=24
            )
            
            duration_minutes = gr.Number(
                label="Flight Duration (Minutes)",
                value=30,
                minimum=0,
                maximum=59
            )
        
        with gr.Row():
            date = gr.Number(
                label="Date",
                value=15,
                minimum=1,
                maximum=31
            )
            
            month = gr.Number(
                label="Month",
                value=6,
                minimum=1,
                maximum=12
            )
            
            year = gr.Number(
                label="Year",
                value=2024,
                minimum=2019,
                maximum=2030
            )
    
    predict_btn = gr.Button("Predict Price", size="lg")
    
    with gr.Row():
        output = gr.Textbox(
            label="Predicted Flight Price",
            placeholder="Click 'Predict Price' to get the prediction",
            interactive=False,
            scale=2
        )
    
    # Set up the prediction function
    predict_btn.click(
        fn=predict_flight_price,
        inputs=[airline, source, destination, duration_hours, duration_minutes, 
                total_stops, date, month, year],
        outputs=output
    )
    
    gr.Markdown("---")
    gr.Markdown("### About")
    gr.Markdown("This model uses Random Forest Regression trained on historical flight data to predict prices. The prediction is based on factors like airline, route, duration, stops, and travel date.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
