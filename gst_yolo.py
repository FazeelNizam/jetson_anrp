import cv2
import numpy as np
import time
import sys

# ================= CONFIGURATION =================
# RTSP URL
rtsp_url = "rtsp://admin:FazNiz!12@192.168.137.55:554/Streaming/Channels/101"

# Your Custom ONNX Model
model_path = "yolov5s_batch1.onnx"

# Model Input Resolution (YOLOv5 standard is usually 640x640)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Thresholds
SCORE_THRESHOLD = 0.5   # Filter low confidence
NMS_THRESHOLD = 0.45    # Remove overlapping boxes
CONFIDENCE_THRESHOLD = 0.45

# Class Names (Update this list to match your custom training!)
# Example: If you trained for plates only, maybe just ['license_plate']
# If you used the default COCO dataset, you need the 80 classes list.
# For now, I'll assume a single class or you can fill this in.
class_list = ['license_plate'] 

# ================= LOAD MODEL (OpenCV DNN) =================
print(f"Loading ONNX model: {model_path}...")
try:
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # ENABLE CUDA (Crucial for Jetson Nano Performance)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("CUDA backend enabled.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure opencv-python is installed and supports CUDA.")
    sys.exit(1)

# ================= PIPELINE =================
# Optimized GStreamer Pipeline (Same as before)
gstreamer_pipeline = (
    f"rtspsrc location='{rtsp_url}' latency=0 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 sync=false"
)

# ================= INITIALIZE VIDEO =================
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    sys.exit(0)

print("Camera connected. Press 'q' to quit.")

# Colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(class_list), 3))

# FPS vars
frame_count = 0
start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame drop.")
        continue

    # Get original shape for scaling back later
    row_height, col_width, _ = frame.shape

    # 1. Preprocess: Create a blob from image (Resize to 640x640, Normalize 1/255)
    #    YOLOv5 expects RGB, so swapRB=True
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    
    # 2. Inference
    net.setInput(blob)
    # The output of YOLOv5 ONNX is typically a huge array: (1, 25200, 5 + NumClasses)
    outputs = net.forward()

    # 3. Post-Processing (Unwrap detections)
    # outputs[0] is the (25200, 85) matrix
    detections = outputs[0]
    
    boxes = []
    confidences = []
    class_ids = []

    # Factors to scale 640x640 output back to original frame size (e.g. 640x480)
    x_factor = col_width / INPUT_WIDTH
    y_factor = row_height / INPUT_HEIGHT

    # Filter detections
    # Each row is: [x_center, y_center, width, height, obj_conf, class_scores...]
    
    # Optimization: Filter by object confidence first to speed up loop
    # We keep rows where object_conf > CONFIDENCE_THRESHOLD
    rows = detections[detections[:, 4] > CONFIDENCE_THRESHOLD]

    for row in rows:
        confidence = row[4]
        
        # Get class scores (skip first 5 elements)
        classes_scores = row[5:]
        
        # Find max score class
        # (If you only have 1 class, this is simpler, but this logic works for any number)
        if len(classes_scores) > 0:
            class_id = np.argmax(classes_scores)
            max_score = classes_scores[class_id]
            
            if max_score > SCORE_THRESHOLD:
                # Calculate coordinates
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                
                # Restore to original image size
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                confidences.append(float(confidence * max_score))
                class_ids.append(class_id)
        else:
            # Fallback for single-class models that might not export class scores correctly
            # (Rare, but happens if exported with specific flags)
            left = int((row[0] - 0.5 * row[2]) * x_factor)
            top = int((row[1] - 0.5 * row[3]) * y_factor)
            width = int(row[2] * x_factor)
            height = int(row[3] * y_factor)
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(0)


    # 4. NMS (Non-Maximum Suppression) to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    # 5. Draw Results
    object_count = 0
    for i in indexes:
        # NMSBoxes returns a list of list in some versions, or flat list in others
        idx = int(i) if not isinstance(i, (list, tuple, np.ndarray)) else int(i[0])
        
        box = boxes[idx]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        class_id = class_ids[idx]
        score = confidences[idx]
        
        object_count += 1

        # Safe label lookup
        label_name = class_list[class_id] if class_id < len(class_list) else f"Class {class_id}"
        label = f"{label_name}: {int(score * 100)}%"

        # Color
        color = colors[class_id % len(colors)]

        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw Label Background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 6. FPS Calculation
    frame_count += 1
    if frame_count >= 30:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLOv5 ONNX", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite(f"capture_{int(time.time())}.png", frame)
        print("Screenshot saved.")

cap.release()
cv2.destroyAllWindows()