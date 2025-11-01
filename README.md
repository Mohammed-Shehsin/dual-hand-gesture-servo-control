Dual-Hand Gesture to Servo Angle Mapping
=========================================

Project Name: dual-hand-gesture-servo-control

Description
-----------
This project implements a real-time dual-hand pinch gesture based control system using OpenCV and MediaPipe. Each hand independently maps thumb-index finger distance to a servo rotation angle between 0° and 180°.

Right hand maps to Servo 1
Left hand maps to Servo 2

When thumb and index finger are touching -> mapped angle approaches 0°.
When fingers separate -> mapped angle increases towards 180°.

Current Stage (Working Now):
- Live camera tracking
- Two-hand detection
- Real-time mapped angle visualization for each hand
- Independent calibration for each hand

Future Scope:
- Serial communication to Arduino or ESP32 to physically rotate 2 servos
- Combine with robotic arm for dual-hand control
- Add GUI sliders for manual override

Controls
--------
`` Q = Quit
Z = Set ZERO for Left hand
X = Set MAX for Left hand
N = Set ZERO for Right hand
M = Set MAX for Right hand ``

Usage
-----
1) pip install opencv-python mediapipe numpy
2) python pinch_to_servo_dual.py
3) Show both hands to camera and pinch thumb + index

Files
-----
dual-hand-gesture-servo-control
README.txt  

Author
------
Shehsin (2025)
