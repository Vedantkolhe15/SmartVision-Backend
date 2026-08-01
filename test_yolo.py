from ultralytics import YOLO

# YOLO model load karo
model = YOLO("yolo11n.pt")

# Image par object detection karo
results = model("uploads/dog.jpg")

# Result save karo
for result in results:
    result.save(filename="detection_result.jpg")

print("Detection completed successfully!")