from flask import Flask, Response, render_template
from flask_cors import CORS
import cv2
import cvzone
from ultralytics import YOLO
import math
import time
from ppeDetection.constant.application import APP_HOST, APP_PORT


app = Flask(__name__)
CORS(app)

# Load the YOLO model
model = YOLO("../PPE-Detection-YOLOv8/yolov8l_train/best.pt")
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 
              'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 
              'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 
              'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']


def get_color(class_name, confidence):
    if confidence > 0.5:
        # Note: OpenCV or cvzone utilize BGR instead of RGB
        if class_name in ["Hardhat", "Mask", "Safety Vest", "Gloves"]:
            return (0, 255, 0)  # Green for correctly detected PPE
        elif class_name in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
            return (0, 0, 255)  # Red for absent PPE 
        elif class_name == "Person":
            return (255, 255, 0)  # Cyan for person
    return (0, 0, 255)  # Default to red if low confidence or other classes

def gen_frames():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW: To specify video source, 0: Default camera; 1 and later: External camera
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(cv2.CAP_PROP_FOURCC, 0x32595559) # CAP_PROP_FOURCC: 4-character code of codec
    cap.set(cv2.CAP_PROP_FPS, 25)            # CAP_PROP_FPS: Frame rate
    
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                currentClass = classNames[cls]
                myColor = get_color(currentClass, conf)

                cvzone.putTextRect(img, f'{currentClass} {conf}', 
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255,255,255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)


        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        (flag, encodedImage) = cv2.imencode('.jpg', img)
        if not flag:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index_live.html')

if __name__ == '__main__':
    app.run(host=APP_HOST, port=APP_PORT, debug=True)