from ultralytics import YOLO
import os

# Load the trained YOLOv8 model
model_path = "runs/detect/run_new_13/weights/best.pt"  
model = YOLO(model_path)

# Set path to evaluation images

image1_path = r"C:\GitHub\AER850_project_3\Project 3 Data\data\evaluation\ardmega.jpg"
image2_path = r"C:\GitHub\AER850_project_3\Project 3 Data\data\evaluation\arduno.jpg"
image3_path = r"C:\GitHub\AER850_project_3\Project 3 Data\data\evaluation\rasppi.jpg"



results1 = model.predict(
    source=image1_path, 
    save=True,                
    save_txt=True,           
    save_conf=True,           
    imgsz=1000,  
line_width=4,                  
)

results2 = model.predict(
    source=image2_path,  
    save=True,              
    save_txt=True,           
    save_conf=True,           
    imgsz=1000,  
line_width=2,                  
)



results3 = model.predict(
    source=image3_path,  
    save=True,              
    save_txt=True,           
    save_conf=True,           
    imgsz=1000,  
line_width=3,               
)


save_dir = model.predictor.save_dir
print(f"Labeled images saved at: {save_dir}")
