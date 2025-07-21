# 📈 Nifty 50 Stock Price Predictor API

A FastAPI-based backend for predicting the next-day closing price of the Nifty 50 index using an LSTM model trained on historical data. Deployed using Docker on [Render](https://render.com).

---

## 🚀 Features

- 📥 Fetches real-time Nifty 50 stock data via `yfinance`
- 🔄 Preprocesses data using a pre-trained `MinMaxScaler`
- 🧠 Predicts the next closing price using a trained LSTM model
- 📦 Fully containerized with Docker
- 🌐 Deployed and live on Render
- 🔍 Built-in Swagger UI for easy API testing

---

## 📁 Project Structure

.
├── main.py # FastAPI entry point
├── Dockerfile # Docker setup
├── requirements.txt # Python dependencies
├── utils/
│ └── data_loader.py # Data fetching & preprocessing
├── models/
│ ├── lstm_stock_model.keras # Trained LSTM model
│ └── scaler.pkl # Pre-trained MinMaxScaler

yaml
Copy
Edit

---

## 📦 Requirements

- Python 3.11
- Docker
- Render account (for deployment)

---

## 🧪 Local Development

### 1. Clone the repo

```bash
git clone https://github.com/your-username/nifty50-stock-predictor.git
cd nifty50-stock-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

### 4. Access the API

Docs: <http://localhost:8000/docs>

Predict: <http://localhost:8000/predict?symbol=^NSEI>

# 🐳 Docker Usage

### 1. Build the Docker image

```bash
docker build -t nifty50-predictor .
```

### 2. Run the container

```bash
docker run -p 8000:8000 nifty50-predictor
```

# 🌐 Deployment on Render

Push code to GitHub

Go to <https://render.com>

Create a new Web Service → connect GitHub repo

Set:

Environment: Docker

Port: 8000

Click Create Web Service

Render will automatically build and host the service.

🔗 API Endpoints
Method Endpoint Description
GET / Root health check
GET /predict Predict next-day closing price

Example: /predict

# 📊 Tech Stack

FastAPI – API framework

TensorFlow / Keras – LSTM model

scikit-learn – Scaling

yfinance – Stock data

Docker – Containerization

Render – Cloud deployment
