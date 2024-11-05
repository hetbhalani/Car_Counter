from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while