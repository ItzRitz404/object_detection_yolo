from ultralytics import YOLO

model = YOLO("/home/ritz/Documents/object_detection/runs/detect/train2/weights/best.pt") # or "yolov8n.pt"

# results = model.predict("/home/ritz/Documents/object_detection/dataset/images/train/__2i0wrlec9g3uwX05QDig.jpg", device=0)

results = model.predict(
    source="/home/ritz/Documents/object_detection/images/",
    save=True,
    device=0
)

# print(results)

