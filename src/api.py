import os

import joblib
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Latency Prediction API",
    description="API for predicting latency based on telecom and weather data",
    version="1.0.0"
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model
model_path = os.path.join(os.getcwd(), "models", "best_model.joblib")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Load the preprocessor
preprocessor_path = os.path.join(os.getcwd(), "models", "preprocessor.joblib")
if os.path.exists(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
else:
    preprocessor = None

# Input model for API predictions
class TelecomData(BaseModel):
    PCell_RSRP_max: float
    PCell_RSRQ_max: float
    PCell_RSSI_max: float
    PCell_SNR_max: float
    PCell_Downlink_Average_MCS: float
    PCell_Downlink_bandwidth_MHz: float
    device: str
    speed_kmh: float
    temperature: float
    humidity: float
    area: str

# Default values for dashboard form
default_values = {
    "PCell_RSRP_max": -85.0,
    "PCell_RSRQ_max": -12.0,
    "PCell_RSSI_max": -50.0,
    "PCell_SNR_max": 16.5,
    "PCell_Downlink_Average_MCS": 23.0,
    "PCell_Downlink_bandwidth_MHz": 20.0,
    "device": "pc1",
    "speed_kmh": 30.0,
    "temperature": 25.0,
    "humidity": 0.6,
    "area": "Residential"
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with the prediction form"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_values": default_values,
            "device_options": ["pc1", "pc2", "pc3"],
            "area_options": ["Residential", "Urban", "Rural", "Industrial", "Commercial"]
        }
    )

@app.post("/predict")
async def predict_throughput(data: TelecomData):
    """Make a prediction using the loaded model via API"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Fill other required columns with median/default values
        # Create a function to get dummy data for columns not provided
        filled_data = _fill_missing_columns(input_data)
        
        # Make prediction
        prediction = model.predict(filled_data)[0]
        
        return {
            "prediction": float(prediction),
            "unit": "Mbps",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    PCell_RSRP_max: float = Form(...),
    PCell_RSRQ_max: float = Form(...),
    PCell_RSSI_max: float = Form(...),
    PCell_SNR_max: float = Form(...),
    PCell_Downlink_Average_MCS: float = Form(...),
    PCell_Downlink_bandwidth_MHz: float = Form(...),
    device: str = Form(...),
    speed_kmh: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    area: str = Form(...)
):
    """Make a prediction using form data and render the result"""
    if model is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "default_values": default_values,
                "device_options": ["pc1", "pc2", "pc3"],
                "area_options": ["Residential", "Urban", "Rural", "Industrial", "Commercial"],
                "error": "Model not loaded. Please train the model first."
            }
        )
    
    try:
        # Create a TelecomData object from form data
        data = TelecomData(
            PCell_RSRP_max=PCell_RSRP_max,
            PCell_RSRQ_max=PCell_RSRQ_max,
            PCell_RSSI_max=PCell_RSSI_max,
            PCell_SNR_max=PCell_SNR_max,
            PCell_Downlink_Average_MCS=PCell_Downlink_Average_MCS,
            PCell_Downlink_bandwidth_MHz=PCell_Downlink_bandwidth_MHz,
            device=device,
            speed_kmh=speed_kmh,
            temperature=temperature,
            humidity=humidity,
            area=area
        )
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Fill missing columns
        filled_data = _fill_missing_columns(input_data)
        
        # Make prediction
        prediction = model.predict(filled_data)[0]
        
        # Update default values with current input
        current_values = data.dict()
        
        # Return the response
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "default_values": current_values,
                "device_options": ["pc1", "pc2", "pc3"],
                "area_options": ["Residential", "Urban", "Rural", "Industrial", "Commercial"],
                "prediction": f"{prediction:.2f} Mbps",
                "input_data": current_values
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "default_values": default_values,
                "device_options": ["pc1", "pc2", "pc3"],
                "area_options": ["Residential", "Urban", "Rural", "Industrial", "Commercial"],
                "error": str(e)
            }
        )

def _fill_missing_columns(input_data):
    """Fill missing columns with default/median values"""
    # Default values for columns that might be missing
    defaults = {
        "PCell_Downlink_Num_RBs": 35000.0,
        "PCell_Cell_Identity": 100.0,
        "PCell_freq_MHz": 2100.0,
        "SCell_RSRP_max": -90.0,
        "SCell_RSRQ_max": -15.0,
        "SCell_RSSI_max": -55.0,
        "SCell_SNR_max": 10.0,
        "SCell_Downlink_Num_RBs": 15000.0,
        "SCell_Downlink_Average_MCS": 20.0,
        "SCell_Downlink_bandwidth_MHz": 10.0,
        "SCell_Cell_Identity": 200.0,
        "SCell_freq_MHz": 1800.0,
        "operator": 1,
        "Latitude": 45.0,
        "Longitude": 5.0,
        "Altitude": 100.0,
        "COG": 30.0,
        "precipIntensity": 0.0,
        "precipProbability": 0.0,
        "apparentTemperature": 25.0,
        "dewPoint": 10.0,
        "pressure": 1013.0,
        "windSpeed": 2.0,
        "cloudCover": 0.5,
        "uvIndex": 4.0,
        "visibility": 16.0,
        "Traffic Jam Factor": 0.4
    }
    
    # One-hot encode categorical variables
    if 'device' in input_data.columns:
        input_data = pd.get_dummies(input_data, columns=['device'], prefix='device')
    
    if 'area' in input_data.columns:
        input_data = pd.get_dummies(input_data, columns=['area'], prefix='area')
    
    # Get expected columns from model
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    
    
    # Add missing columns
    for feature in expected_features:
        if feature not in input_data.columns:
            if feature in defaults:
                input_data[feature] = defaults[feature]
            elif feature.startswith('device_'):
                # For one-hot encoded device columns
                device_type = feature.split('_')[1]
                if 'device' in input_data.columns and input_data['device'].iloc[0] == device_type:
                    input_data[feature] = 1
                else:
                    input_data[feature] = 0
            elif feature.startswith('area_'):
                # For one-hot encoded area columns
                area_type = feature.split('_')[1]
                if 'area' in input_data.columns and input_data['area'].iloc[0] == area_type:
                    input_data[feature] = 1
                else:
                    input_data[feature] = 0
            else:
                input_data[feature] = 0  # Default to 0 for unknown columns
    
    # Remove extra columns not needed by the model
    columns_to_drop = [col for col in input_data.columns if col not in expected_features]
    if columns_to_drop:
        input_data = input_data.drop(columns=columns_to_drop)
    
    # Ensure columns are in the right order
    input_data = input_data[expected_features]
    
    return input_data

if __name__ == "__main__":
    import uvicorn

    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
