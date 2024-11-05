from ultralytics import YOLO
import cv2

model = YOLO('./Yolo-Weights/yolov8n.pt')
results = model("../imgs/sch_bus.jpg", show=True)
cv2.waitKey(0)