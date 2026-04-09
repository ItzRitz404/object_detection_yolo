from ultralytics import YOLO

# model = YOLO("yolo11n.pt") # or "yolov8n.pt"
# results = model.train(
# data="/home/ritz/Documents/object_detection/dataset/data.yaml",
# epochs=10,
# imgsz=640,
# batch=16,
# device=0
# )

model = YOLO("/home/ritz/Documents/object_detection/runs/detect/train2/weights/last.pt") # or "yolov8m.pt"

results = model.train(
    resume=True
    )

# results = model.train(
#     data="/u01/yolo_project/dataset/data.yaml",
#     epochs=5,
#     imgsz=640,
#     name="street_signs_mapillary")

# results = model.train(
#     data="/home/ritz/Documents/object_detection/dataset/data.yaml",
#     epochs=100,         
#     imgsz=896,
#     batch=-1,         # auto batch
#     workers=8,
#     cache="disk",
#     device=0
# )