# Iris and Pupil Tracker

This project uses OpenCV and MediaPipe to track the iris and pupil of a person in real-time using a webcam. The program detects the iris and pupil, measures their sizes, and calculates the ratio between them for both eyes. The processed visuals are displayed on the screen with annotated information.

---

## Features
- Detects iris and pupil positions for both eyes.
- Measures the radius of the iris and pupil.
- Calculates the ratio of iris to pupil radius.
- Annotates the detected features on the webcam feed.

---

## Prerequisites
Ensure you have the following installed in your Python environment:
- Python 3.8 or later
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

---

## Installation
1. Clone the repository or download the code.
2. Install the required dependencies using pip:
   ```bash
   pip install opencv-python mediapipe numpy
## How to Run
1. Connect a webcam to your system.
2. Run the script:
   ```bash
   python iris_pupil_tracker.py
