import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.clv_engine import train_and_process_clv, predict_single_customer_clv, get_clv_tier_insights, MODEL_METRICS

app = FastAPI(
    title="Customer Lifetime Value (CLV) Supervised ML Prediction Platform",
    description="Production-Grade CLV Prediction System using Random Forest / XGBoost Supervised Regression",
    version="2.0.0"
)

CURRENT_CLV_DF = pd.DataFrame()
CURRENT_TRANS_DF = pd.DataFrame()
DATA_PATH = "data/sample_clv_transactions.csv"

def load_default_dataset():
    global CURRENT_CLV_DF, CURRENT_TRANS_DF
    if not os.path.exists(DATA_PATH):
        from data.generate_dataset import generate_clv_transactions
        generate_clv_transactions(output_file=DATA_PATH)
        
    CURRENT_TRANS_DF = pd.read_csv(DATA_PATH)
    CURRENT_CLV_DF = train_and_process_clv(CURRENT_TRANS_DF)

@app.on_event("startup")
def startup_event():
    load_default_dataset()

class CustomerInput(BaseModel):
    customer_id: str
    recency: int
    frequency: int
    monetary: float
    customer_age: int = 365

@app.get("/api/summary")
def get_summary():
    global CURRENT_CLV_DF
    if CURRENT_CLV_DF.empty:
        load_default_dataset()
    if CURRENT_CLV_DF.empty:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
        
    total_customers = int(len(CURRENT_CLV_DF))
    total_historical_revenue = float(CURRENT_CLV_DF['Monetary'].sum())
    total_predicted_clv = float(CURRENT_CLV_DF['Predicted_CLV'].sum())
    avg_historical_spend = float(CURRENT_CLV_DF['Monetary'].mean())
    avg_predicted_clv = float(CURRENT_CLV_DF['Predicted_CLV'].mean())
    
    tier_counts = CURRENT_CLV_DF['CLV_Tier'].value_counts().to_dict()
    tier_revenue = CURRENT_CLV_DF.groupby('CLV_Tier')['Predicted_CLV'].sum().to_dict()
    tier_recency = CURRENT_CLV_DF.groupby('CLV_Tier')['Recency'].mean().to_dict()
    tier_frequency = CURRENT_CLV_DF.groupby('CLV_Tier')['Frequency'].mean().to_dict()
    
    tier_distribution = []
    for tier, count in tier_counts.items():
        tier_distribution.append({
            "tier": tier,
            "count": int(count),
            "percentage": round((count / total_customers) * 100, 2),
            "total_predicted_clv": round(tier_revenue.get(tier, 0.0), 2),
            "avg_recency": round(tier_recency.get(tier, 0.0), 1),
            "avg_frequency": round(tier_frequency.get(tier, 0.0), 1),
            "avg_predicted_clv": round(tier_revenue.get(tier, 0.0) / count, 2) if count > 0 else 0.0
        })
        
    from backend.clv_engine import MODEL_METRICS
    return {
        "total_customers": total_customers,
        "total_historical_revenue": round(total_historical_revenue, 2),
        "total_predicted_clv": round(total_predicted_clv, 2),
        "avg_historical_spend": round(avg_historical_spend, 2),
        "avg_predicted_clv": round(avg_predicted_clv, 2),
        "model_metrics": MODEL_METRICS,
        "tier_distribution": tier_distribution,
        "tier_insights": get_clv_tier_insights()
    }

@app.get("/api/segments")
def get_segments_detail():
    global CURRENT_CLV_DF
    if CURRENT_CLV_DF.empty:
        load_default_dataset()
    insights = get_clv_tier_insights()
    
    tier_data = []
    for tier, group in CURRENT_CLV_DF.groupby('CLV_Tier'):
        tier_data.append({
            "tier": tier,
            "customer_count": int(len(group)),
            "total_historical_spend": round(float(group['Monetary'].sum()), 2),
            "total_predicted_clv": round(float(group['Predicted_CLV'].sum()), 2),
            "avg_recency": round(float(group['Recency'].mean()), 1),
            "avg_frequency": round(float(group['Frequency'].mean()), 1),
            "avg_predicted_clv": round(float(group['Predicted_CLV'].mean()), 2),
            "insights": insights.get(tier, {})
        })
    return tier_data

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global CURRENT_TRANS_DF, CURRENT_CLV_DF
    try:
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
        
        cols = [c.lower() for c in df.columns]
        required_keywords = ['customer', 'date', 'amount']
        missing = [kw for kw in required_keywords if not any(kw in c for c in cols)]
        
        if missing:
            raise HTTPException(
                status_code=400, 
                detail="Dataset missing required columns. Expected columns for Customer ID, Date, and Amount."
            )
            
        CURRENT_TRANS_DF = df
        CURRENT_CLV_DF = train_and_process_clv(df)
        
        return {
            "message": "Dataset uploaded and Supervised ML CLV Model retrained successfully!",
            "total_records": len(df),
            "total_customers": len(CURRENT_CLV_DF)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict-customer")
def predict_customer(data: CustomerInput):
    global CURRENT_CLV_DF
    if CURRENT_CLV_DF.empty:
        load_default_dataset()
        
    result = predict_single_customer_clv(data.recency, data.frequency, data.monetary, data.customer_age, CURRENT_CLV_DF)
    result["CustomerID"] = data.customer_id
    return result

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
