import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path
import cv2
import threading
import time
from datetime import datetime
import numpy as np
import re
import os
from ultralytics import YOLO
import easyocr
import queue # Import queue for thread-safe communication

import sys
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# --- Setup Assets Path ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

ASSETS_PATH = Path(resource_path(r"build/assets/frame0"))

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def get_available_cameras():
    """Detects available cameras by trying indices 0-2 and verifying frame capture."""
    available = []
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            available.append(f"Camera {i}")
        cap.release()
    return available if available else ["No Camera Found"]

class Header(ctk.CTkFrame):
    def __init__(self, parent, username):
        super().__init__(parent, fg_color="transparent", height=80)
        
        try:
            logo_img_data = Image.open(relative_to_assets("image_1.png"))
            self.logo_image = ctk.CTkImage(light_image=logo_img_data, dark_image=logo_img_data, size=(50, 50))
            self.logo_label = ctk.CTkLabel(self, image=self.logo_image, text="")
            self.logo_label.pack(side="left", padx=20)
        except Exception:
            self.logo_label = ctk.CTkLabel(self, text="[LOGO]", text_color="white")
            self.logo_label.pack(side="left", padx=20)

        self.right_box = ctk.CTkFrame(self, fg_color="transparent")
        self.right_box.pack(side="right", padx=20)

        self.time_label = ctk.CTkLabel(self.right_box, text="", font=("Podkova", 20), text_color="white", fg_color="transparent")
        self.time_label.pack(side="left", padx=15)
        
        self.user_label = ctk.CTkLabel(self.right_box, text=username, font=("Podkova", 20, "bold"), text_color="#B7EBDE", fg_color="transparent")
        self.user_label.pack(side="left", padx=15)

        self.update_time()

    def update_time(self):
        now = datetime.now()
        time_str = now.strftime("%I:%M %p  %d/%m/%Y")
        self.time_label.configure(text=time_str)
        self.after(1000, self.update_time)

# class AreaCards(ctk.CTkFrame):
#     def __init__(self, parent, card_id, area_name, total_slots, available_slots, connection_error):
#         super().__init__(parent, fg_color="transparent")


class StreamPanel(ctk.CTkFrame):
    def __init__(self, parent, title, panel_id, available_cams, conflict_callback, model, reader):
        super().__init__(parent, fg_color="transparent")
        self.panel_id = panel_id
        self.conflict_callback = conflict_callback
        self.model = model
        self.reader = reader
        
        self.is_streaming = False
        self.cap = None
        self.save_dir = 'detected_plates'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Queue for thread-safe UI updates
        # Maxsize=2 ensures we don't build up a huge backlog if UI is slow, keeping it real-time
        self.frame_queue = queue.Queue(maxsize=2)
        
        # BBox Colors
        self.bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

        # Title
        self.title_label = ctk.CTkLabel(self, text=title, font=("Podkova", 20, "bold"), text_color="#B7EBDE", anchor="w")
        self.title_label.pack(fill="x", pady=(0, 10))

        # Source & Buttons
        self.source_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.source_frame.pack(fill="x", pady=(0, 10))

        self.rtsp_entry = ctk.CTkEntry(self.source_frame, placeholder_text="RTSP Link", width=200)
        self.rtsp_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.rtsp_entry.bind("<KeyRelease>", self.on_source_change)

        self.cam_options = available_cams
        self.cam_dropdown = ctk.CTkOptionMenu(self.source_frame, values=self.cam_options, width=200, command=self.on_source_change)
        if self.cam_options:
            self.cam_dropdown.set(self.cam_options[0])
        self.cam_dropdown.grid(row=1, column=0, padx=(0, 10), pady=(5, 0), sticky="ew")

        self.stream_btn = ctk.CTkButton(self.source_frame, text="Start Stream", fg_color="#4CCEAC", hover_color="#3ac29b", width=100, command=self.toggle_stream)
        self.stream_btn.grid(row=1, column=1, padx=5, pady=(5, 0))
        
        self.tooltip_label = ctk.CTkLabel(self.source_frame, text="", text_color="#FF5555", font=("Poppins", 10))
        self.tooltip_label.grid(row=2, column=0, columnspan=3, sticky="w")

        # Video Display
        self.video_frame = ctk.CTkFrame(self, corner_radius=10, height=300)
        self.video_frame.pack(fill="both", expand=True)
        self.video_frame.pack_propagate(False) 

        self.video_label = ctk.CTkLabel(self.video_frame, text="", corner_radius=10)
        self.video_label.pack(fill="both", expand=True)

        self.status_indicator = ctk.CTkLabel(self.video_frame, text="● Offline", text_color="#FF5555", anchor="nw", font=("Poppins", 12, "bold"), fg_color="#3A1C1C")
        self.status_indicator.place(x=10, y=10)
        
        self.offline_overlay = ctk.CTkFrame(self.video_frame, fg_color="#3A1C1C", corner_radius=10)
        self.offline_icon = ctk.CTkLabel(self.offline_overlay, text="🚫", font=("Arial", 40), text_color="#FF5555")
        self.offline_icon.place(relx=0.5, rely=0.5, anchor="center")
        
        self.show_offline_state()
        
        # Start the UI update loop
        self.update_ui()

    def on_source_change(self, _=None):
        rtsp_text = self.rtsp_entry.get()
        if rtsp_text:
            self.cam_dropdown.configure(state="disabled")
        else:
            self.cam_dropdown.configure(state="normal")

    def toggle_stream(self):
        if self.is_streaming:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        rtsp = self.rtsp_entry.get()
        if rtsp:
            source = rtsp
        else:
            selection = self.cam_dropdown.get()
            if "No Camera" in selection:
                self.tooltip_label.configure(text="No camera available")
                return
            source = int(selection.split()[-1])

        if not self.conflict_callback(self.panel_id, source):
            self.tooltip_label.configure(text="Stream already in use!")
            return
        
        self.tooltip_label.configure(text="")
        self.is_streaming = True
        self.stream_btn.configure(text="Stop Stream", fg_color="#FF5555", hover_color="#cc4444")
        self.status_indicator.configure(text="● Online", text_color="#4CCEAC", fg_color="transparent")
        self.status_indicator.lift()
        
        self.rtsp_entry.configure(state="disabled")
        self.cam_dropdown.configure(state="disabled")
        self.offline_overlay.place_forget()
        
        self.cap = cv2.VideoCapture(source)
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_stream(self):
        self.is_streaming = False
        self.stream_btn.configure(text="Start Stream", fg_color="#4CCEAC")
        self.status_indicator.configure(text="● Offline", text_color="#FF5555")
        
        self.rtsp_entry.configure(state="normal")
        self.on_source_change() 
        
        if self.cap:
            self.cap.release()
        
        self.show_offline_state()
        self.conflict_callback(self.panel_id, None)

    def show_offline_state(self):
        self.video_label.configure(image=None)
        self.offline_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.status_indicator.configure(fg_color="#3A1C1C")
        self.status_indicator.lift()

    def update_ui(self):
        """Checks the queue for new frames and updates the UI in the main thread."""
        try:
            frame = None
            # Get the latest frame available
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            
            if frame:
                self.video_label.configure(image=frame)
                self.video_label.image = frame
        except Exception:
            pass
        
        # Schedule the next check in 10ms
        if self.winfo_exists():
            self.after(10, self.update_ui)

    def video_loop(self):
        while self.is_streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # --- Detection Logic ---
            try:
                det_thresh = 0.7
                ocr_prob = 0.98
                
                frame_h, frame_w, _ = frame.shape
                results = self.model(frame, verbose=False)
                detections = results[0].boxes
                
                for i in range(len(detections)):
                    conf = detections[i].conf.item()
                    if conf > det_thresh:
                        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                        xmin, ymin, xmax, ymax = xyxy
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(frame_w, xmax), min(frame_h, ymax)
                        
                        classidx = int(detections[i].cls.item())
                        classname = self.model.names[classidx]
                        
                        # OCR
                        ocr_text = ""
                        
                        plate_crop = frame[ymin:ymax, xmin:xmax]
                        if plate_crop.size > 0:
                            img_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
                            gray = cv2.GaussianBlur(gray, (1, 1), 10)
                            structuring_element = np.zeros((40, 40), np.uint8)
                            structuring_element[1:-1, 1:-1] = 1
                            final_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuring_element)
                            
                            ocr_results = self.reader.readtext(final_img, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            
                            for (bbox, text, prob) in ocr_results:
                                if prob >= ocr_prob:
                                    ocr_text += text
                                    ocr_prob = prob
                            
                            ocr_text = ocr_text.strip()
                            
                            if ocr_text and len(ocr_text) >= 6:
                                safe_text = re.sub(r'[^a-zA-Z0-9-]', '', ocr_text)
                                if safe_text:
                                    img_name = os.path.join(self.save_dir, f'stream{self.panel_id}_{safe_text}_{ocr_prob:.2f}.jpg')
                                    cv2.imwrite(img_name, final_img)
                        
                        # Draw
                        color = self.bbox_colors[classidx % 10]
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        
                        label = f'{classname}: {int(conf*100)}%'
                        if ocr_text:
                            label += f' | {ocr_text} | {ocr_prob:.2f}'
                            
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_ymin = max(ymin, labelSize[1] + 10)
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            except Exception as e:
                print(f"Detection Error: {e}")

            # Display Preparation
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            
            # We need to get the label size safely. 
            # Note: winfo_width/height might be 1 if not yet rendered, but usually okay in loop.
            # However, accessing GUI properties from thread is also risky.
            # Better to pass a fixed size or rely on the last known size if possible.
            # For now, we'll try to keep it simple but safe.
            # Ideally, we should resize in the main thread, but resizing is heavy.
            # Let's resize here assuming 640x480 or similar if we can't get size, 
            # OR just send the PIL image and let the main thread resize (might be laggy).
            # COMPROMISE: We will try to read size. If it fails or is 1, use default.
            # But reading size is a GUI op. 
            # Let's just resize to a reasonable fixed size for the dashboard panel to save resources?
            # Or better: send the full PIL image and resize in update_ui? 
            # Resizing in main thread (update_ui) is safer but blocks GUI.
            # Resizing in thread is better for performance.
            # We will assume a fixed reasonable size for the panel or just use the frame size if it fits.
            # Let's stick to the original logic but be careful.
            # Actually, accessing winfo_width() from a thread is NOT thread safe.
            # We should probably resize to a fixed target or just send the image.
            # Let's send the image and resize in update_ui to be 100% thread safe, 
            # even if it costs a bit of main thread CPU.
            
            # However, to avoid main thread lag, we can resize here to a fixed "preview" size 
            # that we know fits the UI, e.g., 640x360.
            # The original code did: lbl_w = self.video_label.winfo_width()
            # We will skip that dynamic resize in the thread and do it in update_ui.
            
            # Put in queue
            if not self.frame_queue.full():
                self.frame_queue.put(img)
            else:
                try:
                    self.frame_queue.get_nowait() # Drop old
                    self.frame_queue.put(img)
                except queue.Empty:
                    pass
            
            time.sleep(0.01)

class DashboardPage(ctk.CTkFrame):
    def __init__(self, parent, controller, username):
        super().__init__(parent, fg_color="#141B2F")
        self.controller = controller
        self.active_streams = {}
        
        # Init Models
        print("Initializing Models...")
        self.model = YOLO("best_ncnn_model")
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("Models Initialized.")
        
        # Header
        self.header = Header(self, username)
        self.header.pack(fill="x", pady=(10, 20))

        # Content Grid
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=20, pady=(20, 5))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=1)

        # Global Controls
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.pack(fill="x", padx=20, pady=(10, 20))

        self.available_cams = get_available_cameras()

        self.stream1 = StreamPanel(self.content, "Camera Stream 01", 1, self.available_cams, self.check_conflict, 
                                   self.model, self.reader)
        self.stream1.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.stream2 = StreamPanel(self.content, "Camera Stream 02", 2, self.available_cams, self.check_conflict,
                                   self.model, self.reader)
        self.stream2.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    def check_conflict(self, panel_id, source):
        if source is None:
            if panel_id in self.active_streams:
                del self.active_streams[panel_id]
            return True
        for pid, src in self.active_streams.items():
            if pid != panel_id and src == source:
                return False
            
        self.active_streams[panel_id] = source
        return True


# Main App --------------------------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Window Setup
        self.title("EntrySync")
        self.geometry("1100x620")
        self.resizable(False, False)
        
        # Configure the grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.current_frame = None
        self.show_dashboard()

    def show_dashboard(self, username="User"):
        if self.current_frame:
            self.current_frame.destroy()
            
        self.current_frame = DashboardPage(self, self, username)
        self.current_frame.grid(row=0, column=0, sticky="nsew")

if __name__ == "__main__":
    app = App()
    app.mainloop()
