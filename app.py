from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ['*']

app = FastAPI(title='Calories Burnt Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=['*'],
   allow_headers=['*']
)

# Cargar modelo
model = load(pathlib.Path('model/calories-v1.joblib'))

# Entradas (ajusta según las columnas de tu dataset excepto 'Calories')
class InputData(BaseModel):
    Gender:int=1   # male=1, female=0
    Age:int=68
    Height:float=190.0
    Weight:float=94.0
    Duration:float=29.0
    Heart_Rate:float=105.0
    Body_Temp:float=40.8

# Salida
class OutputData(BaseModel):
    calories:float=231.0

@app.post('/predict', response_model=OutputData)
def predict(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)[0]  # aquí usamos .predict(), no .predict_proba()
    return {'calories': result}

