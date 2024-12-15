from ultralytics import YOLO


# Initialize YOLOv8 model with pretrained weights
model = YOLO('yolov8n.pt')  


model.train(
    data=r"C:\GitHub\AER850_project_3\Project 3 Data\data\data.yaml",
    epochs=198,
    imgsz=1500,  
    batch=2,   
    workers=0,
    cache=False,
    name="run_new_13"
)


