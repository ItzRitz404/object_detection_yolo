from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()

model = YOLO("yolov8n.pt")

ALLOWED_TYPES = {"image/jpeg", "image/png"}

# @app.post("/test")
# async def test(file: UploadFile = File(...)):
#     if file.content_type not in ALLOWED_TYPES:
#         raise HTTPException(
#             status_code=415,
#             detail=f"Unsupported file type: {file.content_type}. Upload a JPG or PNG."
#         )
    
#     data = await file.read() 
#     if not data:
#         raise HTTPException(status_code=400, detail="No file uploaded")
    
#     try:
#         image = Image.open(io.BytesIO(data)).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
#     image_array = np.array(image)
    
#     results = model.predict(image_array)[0]
    
    
#     detections = []
#     for result in results.boxes:
#         detections.append({
#             "name": results.names[int(result.cls[0])],
#             "Confidence": float(result.conf[0]),
#             "box": {
#                 "x1": float(result.xyxy[0][0]),
#                 "y1": float(result.xyxy[0][1]),
#                 "x2": float(result.xyxy[0][2]),
#                 "y2": float(result.xyxy[0][3])
#             }
#         })
        
#     # returnd JSON response with detections
#     return {
#         "detections": detections
#     }


import base64
from PIL import Image
import io
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO

class ImagePayload(BaseModel):
    img: str

def decode(data: str) -> Image.Image:
    if data.startswith("data:"):
        base = data.split(",", 1)
        if len(base) != 2:
            raise ValueError("Invalid data URI format")
        data = base[1]
        
    # removes all whitespace characters 
    data = "".join(data.split())
    # fixes padding
    data += "=" * (-len(data) % 4)
    
    try:
        decoded_data = base64.b64decode(data, validate=True)
        image = Image.open(io.BytesIO(decoded_data))
        
        # Ensure the image is fully loaded
        image.load()          
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")
    
@app.post("/detect")
async def detect(payload: ImagePayload):
    try:
        image = decode(payload.img)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    image_array = np.array(image)
    results = model.predict(image_array)[0]
    
    detections = []
    for result in results.boxes:
        detections.append({
            "name": results.names[int(result.cls[0])],
            "confidence": float(result.conf[0]),
            "box": {
                "x1": float(result.xyxy[0][0]),
                "y1": float(result.xyxy[0][1]),
                "x2": float(result.xyxy[0][2]),
                "y2": float(result.xyxy[0][3])
            }
        })
        
    return {
        "detections": detections
    }