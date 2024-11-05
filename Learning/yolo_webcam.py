from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model = YOLO("../")

while True:
    success, img = cap.read()
    cv2.imshow("image",img)
    cv2.waitKey(1)