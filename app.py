import os
import time
import requests
import cv2
import depthai as dai
from datetime import datetime

# --- NEW: Import YOLO from ultralytics ---
from ultralytics import YOLO

# DB API Endpoint
DB_API_ENDPOINT = "https://localhost:7123/api/garbage"

# --- NEW: Load your local YOLOv8 model once ---
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

# --- NEW: Class names (update if needed) ---
class_names = [
    'battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_gallon', 'chemical_spray_can', 
    'light_bulb', 'paint_bucket', 'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 'plastic_box',
    'plastic_cup', 'plastic_cup_lid', 'plastic_spoon', 'scrap_paper', 'scrap_plastic', 'snack_bag', 'stick',
    'straw', 'toilet_cleaner'
]

categories = {
    'plastic': [],
    'cardboard': [],
    'chemical': [],
    'can': [],
    'other': []
}

for item in class_names:
    if 'plastic' in item:
        categories['plastic'].append(item)
    elif 'can' in item:
        categories['can'].append(item)
    elif 'cardboard' in item:
        categories['cardboard'].append(item)
    elif 'chemical' in item:
        categories['chemical'].append(item)
    else:
        categories['other'].append(item)

def deleteImageOlderThanDays(days):
    now = datetime.now()
    cutoff = now.timestamp() - (days * 24 * 60 * 60)
    for filename in os.listdir():
        if filename.startswith('image_') and filename.endswith('.jpg'):
            file_time = os.path.getmtime(filename)
            if file_time < cutoff:
                os.remove(filename)
                print(f"Deleted old image: {filename}")
    print(f"Finished deleting images older than {days} days.")


def capture_image():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    xout = pipeline.createXLinkOut()
    xout.setStreamName("image")
    cam_rgb.video.link(xout.input)
    with dai.Device(pipeline) as device:
        print("Calibrating exposure/white balance...")
        q = device.getOutputQueue(name="image", maxSize=1, blocking=True)
        for _ in range(5):
            q.get()
        print("Calibration done.")
        frame = q.get().getCvFrame()
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        return filename

def run_local_inference(image_path):
    results = model.predict(image_path)
    predictions = []
    result = results[0]
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        conf = float(box.conf[0].item())
        predictions.append({
            "class": class_name,
            "confidence": conf
        })
    result.show()
    return {"predictions": predictions}

def get_category(class_name):
    for category, items in categories.items():
        if class_name in items:
            return category
    return "other"

def send_to_api(detected, confidence_score):
    sendData = {
        "detected": detected,
        "confidence_score": confidence_score,
        "cameraId": "d3c1f8b2-4e5f-4a2b-9c3e-8f1b2c3d4e5f",
        "longitude": 4.778720,
        "latitude": 51.591415,
    }
    response = requests.post(DB_API_ENDPOINT, json=sendData, verify=False)
    print(f"Sent to API,  Status: {response.status_code}")
    return response.status_code

deleteImageOlderThanDays(28)

# loop every hour
while True:
    try:
        print("Capturing image...")
        image = capture_image()

        print("Running local AI inference...")
        result = run_local_inference(image)
        print("Result:", result)

        predictions = result.get("predictions", [])
        sent_any = False
        for pred in predictions:
            if pred.get("confidence", 0) > 0.70:
                category = get_category(pred.get("class", "Unknown"))
                send_to_api(category, pred.get("confidence", 0))
                sent_any = True
                print(f"Sent prediction: {category} with confidence {pred.get('confidence', 0)}")
        if not sent_any:
            print("No predictions above confidence threshold.")

    except Exception as e:
        print("Error:", e)

    print("Waiting for 1 hour...")
    time.sleep(10)  # wait for 1 hour
