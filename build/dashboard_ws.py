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
import sys
import json
import asyncio
import websockets
from ultralytics import YOLO
import easyocr
import threading

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# --- Setup Assets Path ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

ASSETS_PATH = Path(resource_path(r"build/assets"))

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def get_available_cameras():
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

        self.time_label = ctk.CTkLabel(self.right_box, text="", font=("Poppins", 20), text_color="white", fg_color="transparent")
        self.time_label.pack(side="left", padx=15)
        
        self.update_time()

    def update_time(self):
        now = datetime.now()
        time_str = now.strftime("%I:%M %p  %d/%m/%Y")
        self.time_label.configure(text=time_str)
        self.after(1000, self.update_time)

class InfoCard(ctk.CTkFrame):
    def __init__(self, parent, title, value, limit=None, is_visitor=False):
        super().__init__(parent, fg_color="#141B2F", corner_radius=10, border_width=1, border_color="#2A3B55")
        self.title_lbl = ctk.CTkLabel(self, text=title, font=("Poppins", 16, "bold"), text_color="#B7EBDE")
        self.title_lbl.pack(pady=(10, 5))
        
        if is_visitor:
            ctk.CTkLabel(self, text="Count", font=("Poppins", 14), text_color="#A0A0A0").pack()
            self.value_lbl = ctk.CTkLabel(self, text=str(value), font=("Poppins", 24, "bold"), text_color="white")
            self.value_lbl.pack(pady=5)
        else:
            ctk.CTkLabel(self, text="Space Available", font=("Poppins", 14), text_color="#A0A0A0").pack()
            self.value_lbl = ctk.CTkLabel(self, text=str(value), font=("Poppins", 24, "bold"), text_color="white")
            self.value_lbl.pack(pady=0)
            ctk.CTkLabel(self, text="Limit", font=("Poppins", 14), text_color="#A0A0A0").pack()
            self.limit_lbl = ctk.CTkLabel(self, text=str(limit), font=("Poppins", 16, "bold"), text_color="white")
            self.limit_lbl.pack(pady=(0, 10))

    def update_values(self, value, limit=None):
        self.value_lbl.configure(text=str(value))
        if limit is not None and hasattr(self, 'limit_lbl'):
            self.limit_lbl.configure(text=str(limit))

class AreaCards(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(0, 15))
        self.park_name_lbl = ctk.CTkLabel(self.header_frame, text="Connecting...", font=("Poppins", 22, "bold"), text_color="#B7EBDE", anchor="w")
        self.park_name_lbl.pack(side="left", padx=10)
        self.capacity_lbl = ctk.CTkLabel(self.header_frame, text="Capacity: -- Slots", font=("Poppins", 18, "bold"), text_color="white", anchor="e")
        self.capacity_lbl.pack(side="right", padx=10)
        self.cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cards_frame.pack(fill="x", expand=True)
        self.cards = {}

    def update_data(self, data):
        self.park_name_lbl.configure(text=data.get("park_name", "Unknown Park"))
        self.capacity_lbl.configure(text=f"Capacity: {data.get('capacity', 0)} Slots")
        passes = data.get("passes", [])
        visitors = data.get("visitors", {})
        
        current_keys = set(self.cards.keys())
        new_keys = set([p['name'] for p in passes] + ['Visitors'])
        
        if current_keys != new_keys:
            for widget in self.cards_frame.winfo_children():
                widget.destroy()
            self.cards = {}
            for p in passes:
                card = InfoCard(self.cards_frame, p['name'], p['available'], p['limit'])
                card.pack(side="left", padx=10, expand=True, fill="both")
                self.cards[p['name']] = card
            if visitors:
                v_card = InfoCard(self.cards_frame, "Visitors", visitors.get('count', 0), is_visitor=True)
                v_card.pack(side="left", padx=10, expand=True, fill="both")
                self.cards['Visitors'] = v_card
        else:
            for p in passes:
                if p['name'] in self.cards:
                    self.cards[p['name']].update_values(p['available'], p['limit'])
            if 'Visitors' in self.cards and visitors:
                self.cards['Visitors'].update_values(visitors.get('count', 0))

class StreamPanel(ctk.CTkFrame):
    def __init__(self, parent, title, panel_id, available_cams, conflict_callback, model, reader, ws_client):
        super().__init__(parent, fg_color="transparent")
        self.panel_id = panel_id
        self.conflict_callback = conflict_callback
        self.model = model
        self.reader = reader
        self.ws_client = ws_client
        
        self.is_streaming = False
        self.cap = None
        self.save_dir = 'detected_plates'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

        self.title_label = ctk.CTkLabel(self, text=title, font=("Poppins", 20, "bold"), text_color="#B7EBDE", anchor="w")
        self.title_label.pack(fill="x", pady=(0, 10))

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

    def video_loop(self):
        while self.is_streaming and self.cap and self.cap.isOpened():
            # Check if inference is paused
            if self.ws_client.inference_paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break
            
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
                                    # PAUSE INFERENCE AND SEND TO SERVER
                                    self.ws_client.send_plate(safe_text)
                                    # Wait for resume (handled by loop check)
                        
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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            lbl_w = self.video_label.winfo_width()
            lbl_h = self.video_label.winfo_height()
            if lbl_w > 1 and lbl_h > 1:
                img = img.resize((lbl_w, lbl_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(lbl_w, lbl_h))
            try:
                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img
            except:
                break
            time.sleep(0.01)

class WebSocketClient:
    def __init__(self, uri, app):
        self.uri = uri
        self.app = app
        self.inference_paused = False
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.start_loop, daemon=True)
        self.thread.start()
        self.websocket = None

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.connect())

    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.uri) as websocket:
                    self.websocket = websocket
                    print("Connected to WebSocket Server")
                    async for message in websocket:
                        await self.handle_message(message)
            except Exception as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(5)

    async def handle_message(self, message):
        data = json.loads(message)
        msg_type = data.get("type")
        
        if msg_type in ["init", "update"]:
            # Update UI in main thread
            self.app.after(0, lambda: self.app.dashboard.area_cards.update_data(data["data"]))
            
        elif msg_type == "verification":
            print(f"Verification Result: {data}")
            # Resume inference
            self.inference_paused = False
            # Show result (could add a toast or log)
            
    def send_plate(self, plate):
        if self.websocket:
            self.inference_paused = True
            print(f"Sending plate: {plate}")
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({"type": "plate_detected", "plate": plate})),
                self.loop
            )

class DashboardPage(ctk.CTkFrame):
    def __init__(self, parent, controller, username, ws_client):
        super().__init__(parent, fg_color="#141B2F")
        self.controller = controller
        self.active_streams = {}
        self.ws_client = ws_client
        
        print("Initializing Models...")
        self.model = YOLO("best_ncnn_model")
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("Models Initialized.")
        
        self.header = Header(self, username)
        self.header.pack(fill="x", pady=(10, 20))

        self.area_cards = AreaCards(self)
        self.area_cards.pack(fill="x", padx=20, pady=(0, 20))

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=20, pady=(0, 5))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=1)

        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.pack(fill="x", padx=20, pady=(10, 20))

        self.available_cams = get_available_cameras()

        self.stream1 = StreamPanel(self.content, "Camera Stream 01", 1, self.available_cams, self.check_conflict, 
                                   self.model, self.reader, self.ws_client)
        self.stream1.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.stream2 = StreamPanel(self.content, "Camera Stream 02", 2, self.available_cams, self.check_conflict,
                                   self.model, self.reader, self.ws_client)
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

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EntrySync")
        self.geometry("1100x780")
        self.resizable(False, True)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Initialize WebSocket Client
        self.ws_client = WebSocketClient("ws://localhost:8765", self)
        
        self.current_frame = None
        self.show_dashboard()

    def show_dashboard(self, username="User"):
        if self.current_frame:
            self.current_frame.destroy()
        self.dashboard = DashboardPage(self, self, username, self.ws_client)
        self.current_frame = self.dashboard
        self.current_frame.grid(row=0, column=0, sticky="nsew")

if __name__ == "__main__":
    app = App()
    app.mainloop()
