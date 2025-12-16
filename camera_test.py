import os
import sys
import argparse
import glob
import time
import re 

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# --- Helper Functions & CUDA Check ---

def open_video_stream(source, width=None, height=None):
    """
    Opens a video stream using GStreamer for RTSP or default backend otherwise.
    """
    cap = None
    if isinstance(source, str) and source.startswith('rtsp://'):
        # GStreamer pipeline for RTSP
        # latency=200 is a good balance.
        gst_pipeline = f"rtspsrc location={source} latency=200 ! decodebin ! videoconvert ! appsink"
        print(f"Attempting to open RTSP stream with GStreamer: {gst_pipeline}")
        try:
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("Successfully opened RTSP stream with GStreamer.")
            else:
                print("Failed to open with GStreamer. Falling back to default backend.")
                cap = None
        except Exception as e:
            print(f"GStreamer error: {e}")
            cap = None

    if cap is None:
        cap = cv2.VideoCapture(source)
    
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
    return cap

# Check for OpenCV CUDA support
USE_CUDA_OPENCV = False
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    if count > 0:
        print(f"OpenCV CUDA module detected. Found {count} GPU(s).")
        USE_CUDA_OPENCV = True
    else:
        print("OpenCV CUDA module not detected. Using CPU.")
except AttributeError:
    print("OpenCV CUDA module not available (AttributeError). Using CPU.")

# -------------------------------------

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")',
                    default=None)

args = parser.parse_args()


# Parse user inputs
model_path = "best_ncnn_model"
img_source = "0"
# img_source = "rtsp://admin:admin1234@192.168.0.10:554/cam/realmonitor?channel=1&subtype=1"
min_thresh = 0.7
user_res = args.resolution

# --- OCR Initialization ---
print("Initializing EasyOCR... (This may take a moment)")
# Set gpu=True for Windows PC. It will use CPU if no compatible GPU is found.
reader = easyocr.Reader(['en'], gpu=True) 

# Create directory for saved plates
save_dir = 'detected_plates'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# -------------------------

# Select rtsp streaming protocol
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load the model into memory and get lablemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input
url_prefixes = ['http://', 'https://', 'rtsp://']
source_type = None
cap_arg = None 

if img_source.isdigit(): 
    source_type = 'camera'
    cap_arg = int(img_source)
elif any(img_source.startswith(prefix) for prefix in url_prefixes): 
    source_type = 'camera' 
    cap_arg = img_source
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Load or initialize image source
w = resW if user_res else None
h = resH if user_res else None
cap = open_video_stream(cap_arg, w, h)

# Set bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

print("Starting inference loop...")
# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from camera
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Unable to read frames from the camera.')
        break
    
    # Get frame dimensions
    if frame is None:
        print(f"Error reading frame from source: {img_source}")
        break
    
    frame_h, frame_w, _ = frame.shape

    # Resize frame to desired display resolution
    if resize == True:
        if USE_CUDA_OPENCV:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_frame = cv2.cuda.resize(gpu_frame, (resW, resH))
                frame = gpu_frame.download()
                frame_h, frame_w, _ = frame.shape
            except Exception as e:
                # print(f"CUDA Resize error: {e}. Falling back to CPU.")
                frame = cv2.resize(frame,(resW,resH))
                frame_h, frame_w, _ = frame.shape
        else:
            frame = cv2.resize(frame,(resW,resH))
            frame_h, frame_w, _ = frame.shape


    # Run inference on frame
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # Go through each detection
    for i in range(len(detections)):
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            
            # Get bounding box coordinates
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            
            # Ensure coordinates are within frame bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame_w, xmax)
            ymax = min(frame_h, ymax)

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # --- OCR PROCESSING START ---
            ocr_text = ""
            ocr_prob = ""
            try:
                # Crop the detected plate
                plate_crop = frame[ymin:ymax, xmin:xmax]
                # Convert to RGB for EasyOCR
                # img = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (1, 1), 10)
                structuring_element = np.zeros((40, 40), np.uint8)
                structuring_element[1:-1, 1:-1] = 1
                final_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuring_element)
                
                # Only run OCR if the crop is valid
                if plate_crop.size > 0:
                    ocr_results = reader.readtext(final_img, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    
                    # Iterate through results and check confidence
                    for (bbox, text, prob) in ocr_results:
                        ocr_prob = round(prob, 2)
                        # Only keep text if probability is higher than your threshold
                        if prob >= 0.98:
                            ocr_text += text
                            
                    ocr_text = ocr_text.strip()

                    # Discard prediction if all characters are not detected
                    if len(ocr_text) < 6:
                        ocr_text = ""

            except Exception as e:
                print(f"OCR Error: {e}")
            # --- OCR PROCESSING END ---

            # --- SAVE LOGIC ---
            if ocr_text:
                # Use regex to keep only alphanumeric for safe filename
                safe_text = re.sub(r'[^a-zA-Z0-9-]', '', ocr_text)
                if safe_text:
                    # Add a timestamp to prevent overwriting files with the same name
                    img_name = os.path.join(save_dir, f'{safe_text}_{ocr_prob}.jpg')
                    cv2.imwrite(img_name, final_img)
            # --- SAVE LOGIC END ---

            # --- DRAWING LOGIC ---
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            
            # Update label to include OCR text if available
            label = f'{classname}: {int(conf*100)}%'
            if ocr_text:
                label += f' | {ocr_text}'
                label += f' | {ocr_prob}'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
            
            object_count = object_count + 1
            # --- DRAWING LOGIC END ---


    # Calculate and draw framerate
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO+OCR detection results',frame) # Display image

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): # Press 'q' to quit
        break
    elif key == ord('s'): # Press 's' to pause
        cv2.waitKey(0)
    elif key == ord('p'): # Press 'p' to save screenshot
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)

    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'camera':
    cap.release()
cv2.destroyAllWindows()