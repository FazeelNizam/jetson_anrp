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

class InfoCard(ctk.CTkFrame):
    def __init__(self, parent, title, value, limit=None, is_visitor=False):
        super().__init__(parent, fg_color="#141B2F", corner_radius=10, border_width=1, border_color="#2A3B55")
        self.title = title
        self.value = value
        self.limit = limit
        self.is_visitor = is_visitor
        
        # Title
        self.title_lbl = ctk.CTkLabel(self, text=title, font=("Podkova", 16, "bold"), text_color="#B7EBDE")
        self.title_lbl.pack(pady=(10, 5))
        
        if is_visitor:
            ctk.CTkLabel(self, text="Count", font=("Poppins", 12), text_color="#A0A0A0").pack()
            self.value_lbl = ctk.CTkLabel(self, text=str(value), font=("Poppins", 24, "bold"), text_color="white")
            self.value_lbl.pack(pady=5)
        else:
            ctk.CTkLabel(self, text="Space Available", font=("Poppins", 12), text_color="#A0A0A0").pack()
            self.value_lbl = ctk.CTkLabel(self, text=str(value), font=("Poppins", 24, "bold"), text_color="white")
            self.value_lbl.pack(pady=0)
            
            ctk.CTkLabel(self, text="Limit", font=("Poppins", 12), text_color="#A0A0A0").pack()
            self.limit_lbl = ctk.CTkLabel(self, text=str(limit), font=("Poppins", 16, "bold"), text_color="white")
            self.limit_lbl.pack(pady=(0, 10))

    def update_values(self, value, limit=None):
        self.value_lbl.configure(text=str(value))
        if limit is not None and hasattr(self, 'limit_lbl'):
            self.limit_lbl.configure(text=str(limit))

class AreaCards(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        
        # Header Row: Park Name and Total Capacity
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(0, 15))
        
        self.park_name_lbl = ctk.CTkLabel(self.header_frame, text="Loading...", font=("Podkova", 22, "bold"), text_color="#B7EBDE", anchor="w")
        self.park_name_lbl.pack(side="left", padx=10)
        
        self.capacity_lbl = ctk.CTkLabel(self.header_frame, text="Capacity: -- Slots", font=("Podkova", 18, "bold"), text_color="white", anchor="e")
        self.capacity_lbl.pack(side="right", padx=10)
        
        # Cards Container
        self.cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cards_frame.pack(fill="x", expand=True)
        
        self.cards = {}
        self.data_file = "parking_data.json"
        
        # Start polling for data
        self.update_data()

    def update_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Update Header
                self.park_name_lbl.configure(text=data.get("park_name", "Unknown Park"))
                self.capacity_lbl.configure(text=f"Capacity: {data.get('capacity', 0)} Slots")
                
                # Update Cards
                passes = data.get("passes", [])
                visitors = data.get("visitors", {})
                
                # Clear old cards if structure changes significantly (simplified for now: just recreate if count differs)
                # Ideally, we update existing widgets.
                
                # Process Passes
                current_keys = set(self.cards.keys())
                new_keys = set([p['name'] for p in passes] + ['Visitors'])
                
                # If structure changed, clear all (simple approach)
                if current_keys != new_keys:
                    for widget in self.cards_frame.winfo_children():
                        widget.destroy()
                    self.cards = {}
                    
                    # Create Pass Cards
                    for p in passes:
                        card = InfoCard(self.cards_frame, p['name'], p['available'], p['limit'])
                        card.pack(side="left", padx=10, expand=True, fill="both")
                        self.cards[p['name']] = card
                        
                    # Create Visitor Card
                    if visitors:
                        v_card = InfoCard(self.cards_frame, "Visitors", visitors.get('count', 0), is_visitor=True)
                        v_card.pack(side="left", padx=10, expand=True, fill="both")
                        self.cards['Visitors'] = v_card
                else:
                    # Update existing
                    for p in passes:
                        if p['name'] in self.cards:
                            self.cards[p['name']].update_values(p['available'], p['limit'])
                    
                    if 'Visitors' in self.cards and visitors:
                        self.cards['Visitors'].update_values(visitors.get('count', 0))
                        
        except Exception as e:
            print(f"Error updating area cards: {e}")
            
        # Poll every 1 second
        self.after(1000, self.update_data)


class StreamPanel(ctk.CTkFrame):
    def __init__(self, parent, title, panel_id, available_cams, conflict_callback):
        super().__init__(parent, fg_color="transparent")
        self.panel_id = panel_id
        self.conflict_callback = conflict_callback
        
        self.is_streaming = False
        self.cap = None
        
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
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Display
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

class DashboardPage(ctk.CTkFrame):
    def __init__(self, parent, controller, username):
        super().__init__(parent, fg_color="#141B2F")
        self.controller = controller
        self.active_streams = {}
        
        # Header
        self.header = Header(self, username)
        self.header.pack(fill="x", pady=(10, 20))

        # Area Cards (Dynamic)
        self.area_cards = AreaCards(self)
        self.area_cards.pack(fill="x", padx=20, pady=(0, 20))

        # Content Grid
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=20, pady=(0, 5))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=1)

        # Global Controls
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.pack(fill="x", padx=20, pady=(10, 20))

        self.available_cams = get_available_cameras()

        self.stream1 = StreamPanel(self.content, "Camera Stream 01", 1, self.available_cams, self.check_conflict)
        self.stream1.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.stream2 = StreamPanel(self.content, "Camera Stream 02", 2, self.available_cams, self.check_conflict)
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
        self.geometry("1100x780") # Increased height for new cards
        self.resizable(False, True) # Allow resize
        
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