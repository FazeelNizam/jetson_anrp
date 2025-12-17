import cv2

# CONFIGURATION
# 1. Use double quotes for the URL string to handle special characters
rtsp_url = "rtsp://admin:FazNiz!12@192.168.137.55:554/Streaming/Channels/101"

# 2. GStreamer Pipeline Definition
# - rtspsrc: Source with 0 latency for real-time speed
# - nvv4l2decoder: Hardware decoding on Jetson Nano
# - nvvidconv: Converts hardware memory (NVMM) to raw video for OpenCV
# - videoconvert: Ensures color format compatibility (BGR)
# - appsink: Passes the frame to Python
gstreamer_pipeline = (
    f"rtspsrc location='rtsp://admin:FazNiz!12@192.168.137.55:554/Streaming/Channels/101' latency=0 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
)

# 3. Initialize Video Capture
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream. Check URL or Network.")
    exit()

print("Camera connected. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame drop or connection lost.")
        break

    # --- INSERT YOLO INFERENCE HERE ---
    # results = model(frame)
    # annotated_frame = results[0].plot()
    # ----------------------------------

    # Display for testing (Remove in production to save FPS)
    cv2.imshow('Hikvision Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()