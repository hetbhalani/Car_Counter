from ultralytics import YOLO
import cv2

model = YOLO('./Yolo-Weights/yolov8l.pt')
results = model("./imgs/sch_bus.jpg", show=True)
cv2.waitKey(0)