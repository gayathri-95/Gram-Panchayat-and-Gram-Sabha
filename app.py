from fastapi import FastAPI
from pydantic import BaseModel
import joblib
myapp = FastAPI()

# Load trained model
model = joblib.load(open("model.pkl","rb"))


# Request schema
class gramPanchayats(BaseModel):
    stateCode: int
    finYear: str
    totalGP: int
    sabhaHeld: int
    gpdpUpload: int

# Home endpoint
@myapp.get("/")
def home():
    return {
        "message": "Gram Panchayats Prediction API Running"
    }
# Prediction endpoint
@myapp.post("/predict")
def predict(data: gramPanchayats):
    print(data.finYear)
    fin_year_encoded = year_mapping.get(data.finYear)
    features = [[
        data.stateCode,
        fin_year_encoded,
        data.totalGP,
        data.sabhaHeld,
        data.gpdpUpload,
    ]]

    prediction = model.predict(features)
    print(prediction)
    print(model.predict_proba(features))
    fullyApproved = {
        0: "Fully Approved",
        1: "Not Fully Approved"
    }

    return {
        "prediction": fullyApproved[int(prediction[0])]
    }
year_mapping = {"2020-2021": 0, "2021-2022": 1, "2022-2023": 2,"2023-2024": 3}
