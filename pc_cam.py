import os
import sys
import argparse
import glob
import time
import re  # Import regex for cleaning filenames

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr  # Import EasyOCR

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source: file, folder, video, camera index ("0"), or IP URL ("rtsp://...")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# --- OCR Initialization ---
print("Initializing EasyOCR... (This may take a moment)")
# Set gpu=True for Windows PC. It will use CPU if no compatible GPU is found.
reader = easyocr.Reader(['en'], gpu=True) 

# Create directory for saved plates
save_dir = 'detected_plates'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# -------------------------

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']
url_prefixes = ['http://', 'https://', 'rtsp://']
source_type = None
cap_arg = None 

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
        cap_arg = img_source
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif img_source.isdigit(): 
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

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','camera']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'camera':
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

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

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': 
        if img_count >= len(imgs_list):
            print('All images have been processed.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video' or source_type == 'camera':
        ret, frame = cap.read()
        if not ret or frame is None:
            if source_type == 'video': print('Reached end of the video file.')
            else: print('Unable to read frames from the camera.')
            break
    
    # Get frame dimensions
    if frame is None:
        print(f"Error reading frame from source: {img_source}")
        break
    
    frame_h, frame_w, _ = frame.shape

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))
        # Update frame dimensions if resized
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
            try:
                # Crop the detected plate
                plate_crop = frame[ymin:ymax, xmin:xmax]
                
                # Only run OCR if the crop is valid
                if plate_crop.size > 0:
                    ocr_results = reader.readtext(plate_crop, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    ocr_text = "".join(ocr_results).strip()
            except Exception as e:
                print(f"OCR Error: {e}")
            # --- OCR PROCESSING END ---

            # --- SAVE LOGIC ---
            if ocr_text:
                # Use regex to keep only alphanumeric for safe filename
                safe_text = re.sub(r'[^a-zA-Z0-9]', '', ocr_text)
                if safe_text:
                    # Add a timestamp to prevent overwriting files with the same name
                    img_name = os.path.join(save_dir, f'{safe_text}_{int(time.time()*100)}.jpg')
                    cv2.imwrite(img_name, plate_crop)
            # --- SAVE LOGIC END ---

            # --- DRAWING LOGIC ---
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            
            # Update label to include OCR text if available
            label = f'{classname}: {int(conf*100)}%'
            if ocr_text:
                label += f' | {ocr_text}'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
            
            object_count = object_count + 1
            # --- DRAWING LOGIC END ---


    # Calculate and draw framerate
    if source_type == 'video' or source_type == 'camera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO+OCR detection results',frame) # Display image
    if record: recorder.write(frame)

    # Handle keypresses
    wait_time = 0 if source_type in ['image', 'folder'] else 5
    key = cv2.waitKey(wait_time) & 0xFF
    
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
if record: recorder.release()
cv2.destroyAllWindows()