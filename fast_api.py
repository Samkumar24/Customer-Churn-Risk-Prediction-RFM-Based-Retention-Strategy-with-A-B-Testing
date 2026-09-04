from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from io import BytesIO

import mlflow
import pandas as pd


# ============================================================
# 1. CREATE FASTAPI APP
# ============================================================

app = FastAPI(title="Churn Prediction API")


# ============================================================
# 2. MLflow CONFIGURATION
# ============================================================

# For local development, this is your local MLflow server.
# Later, when using SageMaker-hosted MLflow, replace this
# with your actual remote MLflow tracking URI.

mlflow.set_tracking_uri("http://localhost:5000")


production_model_name = "Balanced_Random_forest"

model_uri = f"models:/{production_model_name}@champion"

model = mlflow.sklearn.load_model(model_uri)


# ============================================================
# 3. S3 CONFIGURATION
# ============================================================


BUCKET_NAME = "churn-prediction-monitoring"



# 5. CUSTOMER DATA SCHEMA
# ============================================================

class Customer_data(BaseModel):

    country: str
    age: float
    gender: str
    membership_tier: str

    total_orders: float
    total_spend_usd: float
    avg_order_value_usd: float
    days_since_last_purchase: float

    preferred_category: str
    preferred_device: str
    preferred_payment_method: str
    acquisition_channel: str

    reviews_given: float
    avg_review_score: float
    returns_made: float
    wishlist_items: float
    newsletter_subscribed: float


# ============================================================
# 6. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Churn Prediction API is running"
    }


# ============================================================
# 7. MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/model")
def model_info():

    return {
        "model": production_model_name,
        "alias": "champion",
        "status": "loaded"
    }


# ============================================================
# 8. JSON PREDICTION ENDPOINT
# ============================================================

@app.put("/predict")
def prediction_endpoint(data: list[Customer_data]):

    input_data = pd.DataFrame(
        [customer.model_dump() for customer in data]
    )

    probability = model.predict_proba(input_data)

    return {
        "prediction": probability[:, 1].tolist()
    }

# ============================================================
# 9. CSV PREDICTION ENDPOINT
# ============================================================

@app.post("/predict_csv")
async def prediction_csv_endpoint(file: UploadFile = File(...)):

    contents = await file.read()

    try:

        input_data = pd.read_csv(
            BytesIO(contents)
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not parse uploaded file as CSV."
        )


    if "customer_id" not in input_data.columns:

        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'customer_id' column."
        )


    customer_ids = input_data["customer_id"].tolist()
    input_data = input_data.drop(columns=["customer_id"])

    probability = model.predict_proba(input_data)


    return {
        "customer_id": customer_ids,
        "prediction": probability[:, 1].tolist(),
    }
 