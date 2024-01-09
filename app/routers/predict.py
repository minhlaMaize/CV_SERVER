import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from starlette.responses import JSONResponse
from app import model

router = APIRouter(prefix="/predict", tags=["Predict"])

class Input(BaseModel):
    gender: str
    price: int 
    number_of_staying_days:int
    number_of_service: int

@router.post("")
async def handleRequest(
    body: Input,
):
    if body.gender not in ['male', 'female']:
                raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="gender must be male or female"
        )
    if body.price < 0 or body.number_of_service < 0 or body.number_of_staying_days < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Number cannot be negative"
        )
    input = [[1 if body.gender == "male" else 0, body.price / 100, body.number_of_service, body.number_of_service]]
    output = predict(np.array(input))
    return JSONResponse(content={"result": output.tolist()[0]})
def predict(input):
    output = model.predict(input)
    return output