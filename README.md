# Car Counting 🚗



This is a Python-based project that uses the YOLOv8 object detection model and the SORT (Simple Online and Realtime Tracking) algorithm to count vehicles in a video feed. 🎥

## Screensort

![ss of detection](./imgs/detect.jpg)

## Features 🚀

- Real-time vehicle detection and counting 🔍
- Supports various vehicle types (cars, trucks, buses, motorbikes) 🚗🚌🏍️
- Utilizes a custom road mask to focus the detection on the desired region 🛣️
- Tracks individual vehicles across frames using the SORT algorithm 🔍🤖

## Tech used 💻

- Python 3.x
- OpenCV 🖥️
- Ultralytics YOLO 🤖
- SORT (Simple Online and Realtime Tracking) 🔍
- cvzone 🎨

## ⚙️ Installation

Follow these steps to run project :

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hetbhalani/Car_Counter.git
    cd Car_Counter
    ```

2. **Install dependencies** :
    ```bash
   pip install requirements.txt
    ```

3. **Run the app**:
    ```bash
    python CarCounter.py 
    ```

## Contributing 🤝

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. 🙌

## Acknowledgments 🙏

Special thanks to Alex Bewley, the creator of the SORT (Simple Online and Realtime Tracking) algorithm, for his valuable contribution to the field of object tracking.
