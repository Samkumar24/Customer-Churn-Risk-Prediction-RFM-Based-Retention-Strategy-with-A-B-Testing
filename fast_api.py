from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

app = FastAPI(title="Churn Prediction API")

mlflow.set_tracking_uri("http://localhost:5000")

production_model_name = 'Balanced_Random_forest'
model_uri = f"models:/{production_model_name}@champion"

model = mlflow.sklearn.load_model(model_uri)



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

@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is running"
    }


# -----------------------------
# Model information
# -----------------------------

@app.get("/model")
def model_info():
    return {
        "model": "Balanced_Random_forest",
        "alias": "champion",
        "status": "loaded"
    }


@app.put("/predict")
def prediction_endpoint(data: list[Customer_data]):

    input_data = pd.DataFrame([customer.model_dump() for customer in data])

    probability = model.predict_proba(input_data)

    #probability = prediction[:,1]

    return {
        "prediction": probability[:, 1].tolist()
    }