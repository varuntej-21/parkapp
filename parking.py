import json
import cv2
import numpy as np
from collections import defaultdict
import datetime


class ParkingManagement:
    def __init__(self, parking_json_path, slot_line_thickness=2):
        self.json_path = parking_json_path
        self.line_width = slot_line_thickness

        # Load slot annotation polygons and types
        try:
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(self.json_path, "r", encoding=encoding) as f:
                        self.slot_list = json.load(f)
                    print(f"Loaded {len(self.slot_list)} parking slots using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                with open(self.json_path, "r") as f:
                    self.slot_list = json.load(f)
                print(f"Loaded {len(self.slot_list)} parking slots (default encoding)")
        except FileNotFoundError:
            print(f"Warning: {self.json_path} not found. Creating empty slot list.")
            self.slot_list = []
        except Exception as e:
            print(f"Error loading JSON file: {str(e)}")
            self.slot_list = []

        # Drawing colors
        self.available_color = (0, 255, 0)  # Green for available
        self.occupied_color = (0, 0, 255)  # Red for occupied
        self.centroid_color = (255, 0, 255)  # Magenta for centroids

        # Temporal filtering for occupancy stability
        self.slot_history = defaultdict(set)
        self.occupancy_buffer = defaultdict(list)
        self.buffer_size = 5

        # Track reset state for video looping
        self.video_reset_flag = False

    def is_point_in_polygon(self, point, polygon_points):
        """Check if point is inside polygon using OpenCV"""
        pts_array = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(pts_array, point, False) >= 0

    def get_track_center(self, track):
        """Get center point of tracked object"""
        ltrb = track.to_ltrb()
        xc = int((ltrb[0] + ltrb[2]) / 2)
        yc = int((ltrb[1] + ltrb[3]) / 2)
        return (xc, yc)

    def temporal_filter_occupancy(self, slot_idx, is_occupied):
        """Apply temporal filtering to reduce occupancy flickering"""
        self.occupancy_buffer[slot_idx].append(is_occupied)

        if len(self.occupancy_buffer[slot_idx]) > self.buffer_size:
            self.occupancy_buffer[slot_idx].pop(0)

        occupied_count = sum(self.occupancy_buffer[slot_idx])
        return occupied_count > len(self.occupancy_buffer[slot_idx]) // 2

    def reset_occupancy_buffers(self):
        """Reset occupancy buffers when video loops"""
        self.occupancy_buffer.clear()
        self.video_reset_flag = True
        print("Occupancy buffers reset for video looping")

    def process_data(self, frame, tracks):
        # Check if we need to reset occupancy buffers (video looped)
        if self.video_reset_flag:
            self.video_reset_flag = False
            # Don't reset buffers here - let them naturally clear over time

        car_count = 0
        bike_count = 0
        slot_status = []
        current_slot_assignments = [set() for _ in self.slot_list]

        # Process all confirmed tracks
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls_id = track.get_det_class()
            center = self.get_track_center(track)

            # Count vehicles internally (not displayed in main.py anymore)
            if cls_id == 2:
                car_count += 1
            elif cls_id == 3:
                bike_count += 1

            # Assign track to parking slots
            for slot_idx, region in enumerate(self.slot_list):
                if self.is_point_in_polygon(center, region["points"]):
                    current_slot_assignments[slot_idx].add(track_id)
                    break

        # Update slot occupancy with temporal filtering
        occupied_count = 0
        for slot_idx, region in enumerate(self.slot_list):
            has_vehicles = len(current_slot_assignments[slot_idx]) > 0
            is_occupied = self.temporal_filter_occupancy(slot_idx, has_vehicles)
            slot_status.append("occupied" if is_occupied else "available")
            if is_occupied:
                occupied_count += 1

            # Draw slot polygon
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            color = self.occupied_color if is_occupied else self.available_color
            cv2.polylines(frame, [pts_array], isClosed=True, color=color, thickness=self.line_width)

            # Draw slot centroid
            M = cv2.moments(pts_array)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), radius=self.line_width * 2,
                           color=self.centroid_color, thickness=-1)

                slot_type = region.get("type", "unknown")
                cv2.putText(frame, slot_type[:1].upper(), (cx - 5, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ❌ Removed drawing bounding boxes and IDs for vehicles
        # (tracking still works, but not visualized)

        available_count = len(self.slot_list) - occupied_count
        info = {
            "Cars": car_count,  # still counted internally
            "Bikes": bike_count,  # still counted internally
            "Occupied": occupied_count,
            "Available": available_count,
            "SlotStatus": slot_status,
            "TotalSlots": len(self.slot_list)
        }
        return frame, info


# -------------------------------------
# PARKING SLOT SELECTOR GUI
class ParkingPtsSelection:
    def __init__(self):
        try:
            from ultralytics.utils import checks
            checks.check_requirements("tkinter")
        except:
            pass

        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox

        self.setup_ui()
        self.initialize_properties()
        self.master.mainloop()

    def setup_ui(self):
        self.master = self.tk.Tk()
        self.master.title("Parking Slot Selector - Enhanced")
        self.master.resizable(False, False)

        # Main canvas
        self.canvas = self.tk.Canvas(self.master, bg="white")
        self.canvas.pack(side=self.tk.BOTTOM)

        # Control frame
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP, padx=10, pady=5)

        main_buttons = [
            ("Upload Image", self.upload_image),
            ("Remove Last Box", self.remove_last_bounding_box),
            ("Clear All", self.clear_all_boxes),
            ("Save JSON", self.save_to_json),
            ("Save As", self.save_as_json),
            ("Load JSON", self.load_from_json)
        ]
        for text, cmd in main_buttons:
            self.tk.Button(button_frame, text=text, command=cmd, padx=10).pack(side=self.tk.LEFT, padx=2)

        type_frame = self.tk.Frame(self.master)
        type_frame.pack(side=self.tk.TOP, pady=5)

        self.tk.Label(type_frame, text="Slot Type:").pack(side=self.tk.LEFT)
        self.type_var = self.tk.StringVar(value="car")

        for slot_type in ["car", "bike"]:
            self.tk.Radiobutton(type_frame, text=slot_type.capitalize(),
                                variable=self.type_var, value=slot_type).pack(side=self.tk.LEFT)

        self.status_label = self.tk.Label(self.master, text="Upload an image to start",
                                          relief=self.tk.SUNKEN, anchor=self.tk.W)
        self.status_label.pack(side=self.tk.BOTTOM, fill=self.tk.X)

    def initialize_properties(self):
        self.image = None
        self.canvas_image = None
        self.rg_data = []  # List of bounding boxes
        self.current_box = []  # Points for current box
        self.types = []  # Slot types

        self.imgw = 0
        self.imgh = 0
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.canvas_max_width = 1200
        self.canvas_max_height = 800

    # (All GUI methods: upload_image, clicks, save/load JSON remain unchanged)
    def upload_image(self):
        from PIL import Image, ImageTk

        img_path = self.filedialog.askopenfilename(
            title="Select parking lot image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not img_path:
            return

        try:
            self.image = Image.open(img_path)
            self.imgw, self.imgh = self.image.size

            # Calculate scaling to fit canvas
            aspect_ratio = self.imgw / self.imgh

            if aspect_ratio > 1:  # Wider than tall
                canvas_width = min(self.canvas_max_width, self.imgw)
                canvas_height = int(canvas_width / aspect_ratio)
            else:  # Taller than wide
                canvas_height = min(self.canvas_max_height, self.imgh)
                canvas_width = int(canvas_height * aspect_ratio)

            # Store scaling factors
            self.scale_x = self.imgw / canvas_width
            self.scale_y = self.imgh / canvas_height

            self.canvas.config(width=canvas_width, height=canvas_height)
            self.canvas_image = ImageTk.PhotoImage(
                self.image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            )
            self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)

            # Bind click events
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.canvas.bind("<Button-3>", self.on_right_click)  # Right click to cancel current box

            # Reset data
            self.rg_data.clear()
            self.current_box.clear()
            self.types.clear()

            self.update_status(f"Image loaded: {self.imgw}x{self.imgh}. Click 4 points to create parking slots.")

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def on_canvas_click(self, event):
        if not self.image:
            return

        self.current_box.append((event.x, event.y))

        # Draw point
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                                fill="red", outline="darkred", width=2)

        # Draw lines between points
        if len(self.current_box) > 1:
            prev_point = self.current_box[-2]
            self.canvas.create_line(prev_point[0], prev_point[1], event.x, event.y,
                                    fill="blue", width=2)

        self.update_status(f"Point {len(self.current_box)}/4 added. Current slot type: {self.type_var.get()}")

        # Complete the box when 4 points are selected
        if len(self.current_box) == 4:
            # Close the polygon
            first_point = self.current_box[0]
            self.canvas.create_line(event.x, event.y, first_point[0], first_point[1],
                                    fill="blue", width=2)

            # Store the box and its type
            self.rg_data.append(self.current_box.copy())
            self.types.append(self.type_var.get())

            # Add label to the box
            center_x = sum(p[0] for p in self.current_box) // 4
            center_y = sum(p[1] for p in self.current_box) // 4
            self.canvas.create_text(center_x, center_y, text=self.type_var.get()[:1].upper(),
                                    fill="white", font=("Arial", 10, "bold"))

            self.current_box.clear()
            self.update_status(f"Slot {len(self.rg_data)} created. Total: {len(self.rg_data)} slots")

    def on_right_click(self, event):
        """Cancel current box creation"""
        if self.current_box:
            self.current_box.clear()
            self.redraw_canvas()
            self.update_status("Current box cancelled.")

    def remove_last_bounding_box(self):
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return

        self.rg_data.pop()
        if self.types:
            self.types.pop()
        self.redraw_canvas()
        self.update_status(f"Last box removed. Total: {len(self.rg_data)} slots")

    def clear_all_boxes(self):
        if not self.rg_data:
            self.messagebox.showinfo("Info", "No boxes to clear.")
            return

        if self.messagebox.askyesno("Confirm", "Clear all parking slots?"):
            self.rg_data.clear()
            self.types.clear()
            self.redraw_canvas()
            self.update_status("All boxes cleared.")

    def redraw_canvas(self):
        if not self.canvas_image:
            return

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)

        # Redraw all completed boxes
        for i, box in enumerate(self.rg_data):
            self.draw_completed_box(box, self.types[i] if i < len(self.types) else "unknown")

    def draw_completed_box(self, box, slot_type):
        # Draw polygon
        for i in range(4):
            start = box[i]
            end = box[(i + 1) % 4]
            self.canvas.create_line(start[0], start[1], end[0], end[1],
                                    fill="blue", width=2)

        # Draw points
        for point in box:
            self.canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3,
                                    fill="red", outline="darkred", width=2)

        # Add type label
        center_x = sum(p[0] for p in box) // 4
        center_y = sum(p[1] for p in box) // 4
        self.canvas.create_text(center_x, center_y, text=slot_type[:1].upper(),
                                fill="white", font=("Arial", 10, "bold"))

    def save_to_json(self):
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No parking slots to save.")
            return

        filename = self.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialvalue="bounding_boxes.json"
        )

        if not filename:
            return

        try:
            data = []
            for idx, box in enumerate(self.rg_data):
                # Convert canvas coordinates back to image coordinates
                original_points = [
                    (int(x * self.scale_x), int(y * self.scale_y))
                    for x, y in box
                ]

                entry = {
                    "points": original_points,
                    "type": self.types[idx] if idx < len(self.types) else "unknown"
                }
                data.append(entry)

            with open(filename, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            self.messagebox.showinfo("Success", f"Saved {len(data)} parking slots to {filename}")
            self.update_status(f"Saved {len(data)} slots to {filename}")

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def save_as_json(self):
        """Save current slots as a new JSON file (Save As)"""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No parking slots to save.")
            return

        filename = self.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"parking_slots_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if not filename:
            return

        try:
            data = []
            for idx, box in enumerate(self.rg_data):
                original_points = [
                    (int(x * self.scale_x), int(y * self.scale_y))
                    for x, y in box
                ]
                entry = {
                    "points": original_points,
                    "type": self.types[idx] if idx < len(self.types) else "unknown"
                }
                data.append(entry)

            with open(filename, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            self.messagebox.showinfo("Success", f"Saved {len(data)} parking slots to {filename}")
            self.update_status(f"Saved {len(data)} slots to {filename}")

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def load_from_json(self):
        filename = self.filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load parking slots"
        )

        if not filename:
            return

        try:
            # Try multiple encodings to handle different JSON file formats
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            data = None
            for encoding in encodings:
                try:
                    with open(filename, "r", encoding=encoding) as f:
                        data = json.load(f)
                    print(f"Loaded JSON using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if data is None:
                # If all encodings failed, try without specifying encoding
                with open(filename, "r") as f:
                    data = json.load(f)
                print("Loaded JSON using default encoding")

            if not self.image:
                self.messagebox.showwarning("Warning", "Please load an image first.")
                return

            # Clear current data
            self.rg_data.clear()
            self.types.clear()

            # Load slots and convert to canvas coordinates
            for entry in data:
                original_points = entry["points"]
                canvas_points = [
                    (int(x / self.scale_x), int(y / self.scale_y))
                    for x, y in original_points
                ]
                self.rg_data.append(canvas_points)
                self.types.append(entry.get("type", "unknown"))

            self.redraw_canvas()
            self.update_status(f"Loaded {len(data)} slots from {filename}")

        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to load: {str(e)}")

    def update_status(self, message):
        self.status_label.config(text=message)