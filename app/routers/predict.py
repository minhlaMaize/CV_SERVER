import cv2
from fastapi.responses import JSONResponse
import numpy as np

from fastapi import UploadFile, APIRouter
from app import knn_model, svm_model, pca, scaler
router = APIRouter(prefix="/predict", tags=["Predict"])


@router.post("")
async def handleRequest(
    file: UploadFile,
):
    content = await file.read()
    output = predict(np.fromstring(content, np.uint8))
    return JSONResponse(content={"knn": {
        "class": output[0].tolist()[0],
    }, "svm": {
        "class": output[1].tolist()[0],
    }})
def predict(input):
    input= np.array([extract_feature_vector_with_img(input)])
    input = scaler.transform(input)
    input = pca.transform(input)
    knn_output = knn_model.predict(input)
    svm_output = svm_model.predict(input)
    print(knn_output)
    return [knn_output, svm_output]
def extract_feature_vector_with_img(img):
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR in OpenCV 3.1
    img = cv2.resize(img, (100, 100))
    _, _, G, _ = sobel_filters(img)  # Apply Sobel filter
    feature = G.flatten()  # Flatten the image matrix into a vector
    return feature
def sobel_filters(img):
    Sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    Sy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float32)

    Ix = cv2.filter2D(img, -1, Sx)
    Iy = cv2.filter2D(img, -1, Sy)

    G=np.hypot(Ix,Iy)
    G=G/G.max()*255
    theta=np.arctan2(Iy,Ix)

    return Ix,Iy,G,theta
