from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from utils.data_loader import fetch_stock_data, preprocess_data
from keras.models import load_model

app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://niftyninja.streamlit.app/"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET"],  # Allows only GET method
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
async def predict_stock():
    """
    Fetches the latest stock data for the given symbol.
    """
    data = fetch_stock_data()
    if data.empty:
        return {"error": "No data found for the given symbol."}
    scaler = joblib.load("models/scaler.pkl")
    X = preprocess_data(data, scaler)
    if X is None:
        return {"error": "Not enough data to generate input sequence."}
    model = load_model("models/lstm_stock_model.keras")
    if model is None:
        return {"error": "Model not found."}
    prediction = model.predict(X)
    if prediction is None:
        return {"error": "Prediction failed."}
    predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))
    return {"prediction": float(predicted_price[0][0])}



@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Nifty 50 Predictor API is running."}


