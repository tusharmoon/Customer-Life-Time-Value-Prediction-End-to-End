# Customer Lifetime Value (CLV) Supervised ML Prediction Platform

An end-to-end production-grade Customer Lifetime Value (CLV) prediction application powered by Python, FastAPI, and Scikit-Learn Supervised Regression (Random Forest Regressor).

---

## ⚙️ Setup & Execution Guide

Follow these steps to run the application locally:

### 1. Open Project Folder
Extract `clv_prediction_app.zip` and navigate into the `clv_prediction_app` directory in VS Code or your terminal.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Dataset (Optional)
The application automatically generates `data/sample_clv_transactions.csv` on startup if missing. You can also generate it manually:
```bash
python data/generate_dataset.py
```

### 4. Run the Server (Single Command)
```bash
uvicorn backend.main:app --reload --port 8001
```

### 5. Access the Web Application
Open your browser and go to:
```text
http://localhost:8001
```

---

## 📊 API Endpoints Reference
- `GET /api/summary`: Returns BI metrics, model accuracy metrics ($R^2$, RMSE), and CLV tier distributions.
- `GET /api/segments`: Retrieves breakdown metrics and strategic playbooks across all predicted CLV tiers.
- `POST /api/predict-customer`: Accepts customer metrics (Recency, Frequency, Monetary, Customer Age) and runs real-time regression inference.
- `POST /api/upload`: Uploads a custom transaction CSV dataset and automatically re-engineers features and retrains the model on the fly.
