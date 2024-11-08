from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("./imgs/cars.mp4")


model = YOLO("./Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("./imgs/mask-road.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limit = [400,280,672,280]

count = []
while True:
    
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    
    results = model(imgRegion,stream=True)
    
    detactions = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            
            #this is using cv2
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1), (x2,y2),(0,255,0),3)
            
            w,h = x2-x1, y2-y1
            
            #Confidence
            conf = math.ceil(box.conf[0]*100)/100
            
            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1.3,thickness=2,offset=5)
                # cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detactions = np.vstack((detactions, currentArray))

            
    trackerResult = tracker.update(detactions)
    cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255))
    for res in trackerResult:
        x1,y1,x2,y2,id = res
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(res)
        w,h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale=1.3,thickness=2,offset=5)

        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        
        if limit[0] < cx < limit[2] and limit[1] < cy < limit[1] + 30:
            if count.count(id) == 0: #list ma check krse, jo ekey var na hoi to list ma add krse
                count.append(id)
                cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,255,0))

            
    cvzone.putTextRect(img,f'Count = {len(count)}',(50,50),scale=1.3,thickness=2,offset=5)

    cv2.imshow("image",img)
    cv2.imshow("imgRegion",imgRegion)
    
    if cv2.waitKey(1) == 27:
        break