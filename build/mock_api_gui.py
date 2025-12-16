import customtkinter as ctk
import json
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MockDataGenerator(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mock API Data Generator")
        self.geometry("400x600")
        
        self.data_file = "parking_data.json"
        
        # Scrollable Frame
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Park Info
        ctk.CTkLabel(self.scroll_frame, text="Park Information", font=("Arial", 16, "bold")).pack(pady=10)
        self.park_name = self.create_entry("Park Name", "SLT HQ Car Park")
        self.capacity = self.create_entry("Total Capacity", "100")
        
        # Passes
        ctk.CTkLabel(self.scroll_frame, text="Passes Configuration", font=("Arial", 16, "bold")).pack(pady=10)
        self.passes_entries = []
        self.add_pass_entry("A Pass", "20", "40")
        self.add_pass_entry("B Pass", "18", "20")
        self.add_pass_entry("C Pass", "15", "20")
        self.add_pass_entry("D Pass", "8", "15")
        
        # Visitors
        ctk.CTkLabel(self.scroll_frame, text="Visitors", font=("Arial", 16, "bold")).pack(pady=10)
        self.visitor_count = self.create_entry("Visitor Count", "5")
        
        # Update Button
        self.update_btn = ctk.CTkButton(self, text="Update Dashboard", command=self.save_data, height=50, font=("Arial", 16, "bold"))
        self.update_btn.pack(fill="x", padx=20, pady=20)
        
        # Initial Save
        self.save_data()

    def create_entry(self, label_text, default_val):
        frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        frame.pack(fill="x", pady=5)
        ctk.CTkLabel(frame, text=label_text, width=100, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(frame)
        entry.insert(0, default_val)
        entry.pack(side="right", expand=True, fill="x")
        return entry

    def add_pass_entry(self, name, avail, limit):
        frame = ctk.CTkFrame(self.scroll_frame, border_width=1, border_color="gray")
        frame.pack(fill="x", pady=5, padx=5)
        
        ctk.CTkLabel(frame, text=name, font=("Arial", 12, "bold")).pack(pady=2)
        
        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x")
        ctk.CTkLabel(row1, text="Name:").pack(side="left", padx=5)
        name_entry = ctk.CTkEntry(row1, width=100)
        name_entry.insert(0, name)
        name_entry.pack(side="right", padx=5)
        
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x")
        ctk.CTkLabel(row2, text="Avail:").pack(side="left", padx=5)
        avail_entry = ctk.CTkEntry(row2, width=60)
        avail_entry.insert(0, avail)
        avail_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(row2, text="Limit:").pack(side="left", padx=5)
        limit_entry = ctk.CTkEntry(row2, width=60)
        limit_entry.insert(0, limit)
        limit_entry.pack(side="left", padx=5)
        
        self.passes_entries.append({
            "name": name_entry,
            "avail": avail_entry,
            "limit": limit_entry
        })

    def save_data(self):
        data = {
            "park_name": self.park_name.get(),
            "capacity": int(self.capacity.get() or 0),
            "passes": [],
            "visitors": {
                "count": int(self.visitor_count.get() or 0)
            }
        }
        
        for p in self.passes_entries:
            data["passes"].append({
                "name": p["name"].get(),
                "available": int(p["avail"].get() or 0),
                "limit": int(p["limit"].get() or 0)
            })
            
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)
            print("Data updated successfully!")
        except Exception as e:
            print(f"Error saving data: {e}")

if __name__ == "__main__":
    app = MockDataGenerator()
    app.mainloop()
