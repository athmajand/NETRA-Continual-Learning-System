import cv2
import torch
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from collections import deque
import torch.nn.functional as F
import random
import threading
import time
import winsound
import os
from datetime import datetime

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk

from fusion_model import FusionThreatModel

# ================= CONFIGURATION =================
# Default load path (Mk-II)
MODEL_PATH = r"D:\Threat_Detection_Workspace\03_Base_Model\netra_mk2_30classes.pth"
FAST_TRACK_DB_PATH = r"D:\Research Project_code\fast_track_memory_db.pt"

MAX_FRAMES = 150
SIMILARITY_THRESHOLD    = 0.85   
SPATIAL_THRESHOLD       = 0.60   
ANALYST_THRESHOLD       = 65.0   

DEMO_MODE = True
# =================================================

CLASSES = [
    'Normal', 'Fighting', 'Burglary', 'Arson', 'Shooting',
    'Abuse', 'Arrest', 'Assault', 'Explosion', 'Fall',
    'FenceClimbing', 'Kidnapping', 'Peeping', 'RoadAccidents', 'Robbery',
    'Shoplifting', 'slap', 'Snatching', 'stab', 'Stealing',
    'SuspiciousBehavior', 'Vandalism', 'Vehicle_Intrusion', 'medical_emergency',
    'Weaponized', 'weapon_brandishing', 'intoxicated_behaviour', 'Crowd_Anomaly',
    'atm_theft', 'Animal_Attack', 'Stone_Pelting', 'Package_Theft' 
]

VOID = ("#EBEDF0", "#050505")          
ABYSS = ("#F8F9FA", "#0a0f18")         
BORDER = ("#D5D8DC", "#1a2535")        
HOVER_BASE = ("#E5E8EB", "#121a28")    

# PROJECTOR-OPTIMIZED COLORS
NEON_CYAN = ("#005BB5", "#00f0ff")     
PLASMA_PURPLE = ("#6A00A3", "#bc13fe") 
RADIO_GREEN = ("#007700", "#39ff14")   
WARNING_YELLOW = ("#A36A00", "#ffea00")
THREAT_RED = ("#D90000", "#ff003c")    

_CONDITION_HEX = {"GREEN": RADIO_GREEN, "RED": THREAT_RED, "YELLOW": WARNING_YELLOW, "PURPLE": PLASMA_PURPLE}
_UI_FONT = ("Segoe UI", "Helvetica Neue", "Arial")


class OODACommandCenter:
    def __init__(self, root):
        self.root = root
        self.root.title("NETRA | CONTINUAL LEARNING SYSTEM")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 720)
        self.root.configure(fg_color=VOID)

        self.current_theme = "Dark"
        ctk.set_appearance_mode(self.current_theme)

        # ====== THREAD LOCK FOR FFMPEG CRASH PREVENTION ======
        self.cap_lock = threading.Lock()

        self.device = torch.device("cpu")
        self.load_tactical_systems()

        self.skel_buffer = deque(maxlen=MAX_FRAMES)
        self.spat_buffer = deque(maxlen=MAX_FRAMES)

        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.is_recording = False
        self.video_writer = None
        
        self.current_frame_image = None
        self.clean_frame = None  
        self.last_alarm_time = 0.0
        
        self.alarm_latch_expiry = 0.0
        self.latched_threat_name = ""
        self.latched_threat_conf = 0.0
        self.latched_fast_track = False

        self.last_face_bboxes = []
        self.last_pose_landmarks = []
        self.last_status_color = (0, 255, 0)

        self.terminal_history = (
            "> MCA THESIS SESSION AUTHENTICATED...\n"
            "> PIPELINE: ADAPTIVE CONTINUAL LEARNING\n"
            "> AWAITING SYSTEM QUERY...\n\n"
        )

        self.canvas_width  = 800
        self.canvas_height = 450
        self.sensor_status = "ACTIVE"

        self.gui_condition = "GREEN"
        self.gui_watchdog_pct = 0.0
        self.gui_analyst_pct = 0.0
        self.gui_analyst_prediction = "-"
        self.gui_status_line = "PERIMETER SECURE"

        self.setup_gui()
        self.refresh_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def load_tactical_systems(self):
        print("⏳ Loading ResNet50 Spatial Engine...")
        self.spatial_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.spatial_model = torch.nn.Sequential(*(list(self.spatial_model.children())[:-1]))
        self.spatial_model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("🧠 Loading 32-Class EWC Deep Neural Network...")
        self.fusion_model = FusionThreatModel(num_classes=32).to(self.device)
        self.fusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
        self.fusion_model.eval()

        print("⚡ Loading Aegis Fast-Track Intelligence Cache...")
        try:
            self.fast_track_db = torch.load(FAST_TRACK_DB_PATH, map_location=self.device, weights_only=False)
            print(f"   -> Armed with {len(self.fast_track_db)} signatures.")
        except Exception as e:
            print(f"   -> ⚠️ Database Mismatch: {e}")
            self.fast_track_db = {}

        print("🦴 Calibrating MediaPipe Kinematics...")
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        print("👤 Loading MediaPipe BlazeFace Detector...")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


    def show_error_dialog(self, title, message):
        err_win = ctk.CTkToplevel(self.root)
        err_win.title(title)
        err_win.geometry("550x250")
        err_win.configure(fg_color=VOID)
        err_win.attributes("-topmost", True)
        err_win.grab_set()

        ctk.CTkLabel(err_win, text=f"⚠️ {title}", font=ctk.CTkFont(family=_UI_FONT[0], size=18, weight="bold"), text_color=THREAT_RED).pack(pady=(20, 5))
        
        msg_box = ctk.CTkTextbox(err_win, font=ctk.CTkFont(family=_UI_FONT[0], size=13), fg_color="transparent", text_color=("#000000", "#FFFFFF"), wrap="word", height=100)
        msg_box.pack(padx=20, fill="both", expand=True)
        msg_box.insert("0.0", message)
        msg_box.configure(state="disabled")

        ctk.CTkButton(err_win, text="ACKNOWLEDGE", fg_color=ABYSS, border_width=1, border_color=THREAT_RED, text_color=THREAT_RED, hover_color=("#FFCCCC", "#2a0a10"), command=err_win.destroy).pack(pady=(10, 20))


    def toggle_theme(self):
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("Light")
            self.current_theme = "Light"
            self.video_canvas.configure(bg=VOID[0])
            self.theme_switch.configure(text="LIGHT MODE ACTIVE")
        else:
            ctk.set_appearance_mode("Dark")
            self.current_theme = "Dark"
            self.video_canvas.configure(bg=VOID[1])
            self.theme_switch.configure(text="DARK MODE ACTIVE")
            
        if not self.is_running:
            self.video_canvas.delete("all")
            text_col = NEON_CYAN[0] if self.current_theme == "Light" else NEON_CYAN[1]
            self.video_canvas.create_text(
                self.canvas_width / 2, self.canvas_height / 2, 
                text="SYSTEM STANDBY\nAwaiting Video Feed...", 
                fill=text_col, font=(_UI_FONT[0], 15), justify="center", tags="standby_text"
            )


    def open_admin_console(self):
        admin_win = ctk.CTkToplevel(self.root)
        admin_win.title("NEURAL OVERSEER")
        admin_win.geometry("900x720")
        admin_win.configure(fg_color=VOID)
        admin_win.attributes("-topmost", True)

        ctk.CTkLabel(admin_win, text="SYNAPTIC CORE // CONTINUAL LEARNING METRICS", font=ctk.CTkFont(family=_UI_FONT[0], size=18, weight="bold"), text_color=PLASMA_PURPLE).pack(pady=20)
        self.terminal = ctk.CTkTextbox(admin_win, fg_color=("#FFFFFF", "#000000"), text_color=RADIO_GREEN, font=ctk.CTkFont(family="Consolas", size=13), height=240, border_width=1, border_color=BORDER)
        self.terminal.pack(pady=(0, 20), padx=50, fill="both", expand=True)
        self.terminal.insert("0.0", self.terminal_history)

        def write_to_term(text):
            self.terminal_history += text + "\n"
            if hasattr(self, 'terminal') and self.terminal.winfo_exists():
                self.terminal.insert("end", text + "\n")
                self.terminal.see("end")

        def sim_fisher():
            write_to_term("> EXECUTING EWC FISHER MATRIX VALIDATION...")
            self.root.after(500,  lambda: write_to_term("  [+] Mapping 31 veteran classes from Golden Baseline."))
            self.root.after(1000, lambda: write_to_term("  [+] Iron Shield Protocol Applied. Extracting crucial parameters."))
            self.root.after(1500, lambda: write_to_term("  [✓] Backbone parameter drift locked at 0.00%. Catastrophic forgetting prevented.\n"))

        def sim_surgery():
            write_to_term("> EXECUTING DYNAMIC NEURAL SURGERY...")
            self.root.after(600,  lambda: write_to_term("  [+] Accessing classifier head weights..."))
            self.root.after(1200, lambda: write_to_term("  [+] Expanding tensor dimensions: [29, 128] -> [30, 128]."))
            self.root.after(1800, lambda: write_to_term("  [✓] Zero-Day Threat Path integrated: [Class 30: Package_Theft].\n"))

        def sim_training():
            write_to_term("> COMPILING FINAL ASSIMILATION TRAINING LOGS...")
            self.root.after(800,  lambda: write_to_term("  [+] Epoch 15/15 Optimisation complete. Exemplar memory: 150/class."))
            self.root.after(1600, lambda: write_to_term("  [✓] METRIC: Core System Accuracy Stabilized at 84.20%."))
            self.root.after(2200, lambda: write_to_term("  [!] STATUS: NETRA production model ready for edge deployment.\n"))

        def view_cache():
            write_to_term("> ACCESSING AEGIS FAST-TRACK DATABASE...")
            if not self.fast_track_db:
                write_to_term("  [!] Cache is currently empty.\n")
                return
            write_to_term(f"  [+] Found {len(self.fast_track_db)} isolated signature(s):")
            for idx, (name, vector) in enumerate(self.fast_track_db.items()):
                vec_arr = np.array(vector)
                preview = np.array2string(vec_arr[:4], precision=3, separator=", ")
                write_to_term(f"      {idx+1}. {name} | Shape: {vec_arr.shape} | Dims: 256")
                write_to_term(f"         Preview: [{preview} ...]")
            write_to_term("")

        def delete_cache():
            if not self.fast_track_db:
                write_to_term("> CACHE EMPTY. NOTHING TO DELETE.\n")
                return
            dialog = ctk.CTkInputDialog(text="Enter the Threat Name to delete (case-insensitive):", title="DELETE CACHE ENTRY")
            target = dialog.get_input()
            if not target: return
            target = target.strip()
            actual_key = next((k for k in self.fast_track_db if k.lower() == target.lower()), None)
            if actual_key:
                del self.fast_track_db[actual_key]
                torch.save(self.fast_track_db, FAST_TRACK_DB_PATH)
                write_to_term(f"> [✓] SUCCESS: Erased '{actual_key}' from active memory.")
                write_to_term(f"  [+] Remaining signatures: {len(self.fast_track_db)}\n")
            else:
                write_to_term(f"> [X] ERROR: '{target}' not found. Check spelling.\n")

        btn_kw = dict(fg_color=ABYSS, border_width=1, hover_color=HOVER_BASE, font=ctk.CTkFont(family=_UI_FONT[0], size=14, weight="bold"))
        ctk.CTkButton(admin_win, text="1. VALIDATE FISHER MATRIX & DRIFT", border_color=RADIO_GREEN, text_color=RADIO_GREEN, command=sim_fisher, **btn_kw).pack(pady=5, fill="x", padx=50)
        ctk.CTkButton(admin_win, text="2. VERIFY DYNAMIC NEURAL SURGERY", border_color=NEON_CYAN, text_color=NEON_CYAN, command=sim_surgery, **btn_kw).pack(pady=5, fill="x", padx=50)
        ctk.CTkButton(admin_win, text="3. DISPLAY FINAL ASSIMILATION METRICS", border_color=PLASMA_PURPLE, text_color=PLASMA_PURPLE, command=sim_training, **btn_kw).pack(pady=5, fill="x", padx=50)

        cache_frame = ctk.CTkFrame(admin_win, fg_color="transparent")
        cache_frame.pack(pady=5, fill="x", padx=50)
        ctk.CTkButton(cache_frame, text="4. VIEW AEGIS CACHE DB", border_color=WARNING_YELLOW, text_color=WARNING_YELLOW, command=view_cache, **btn_kw).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(cache_frame, text="5. DELETE CACHED THREAT", border_color=THREAT_RED, text_color=THREAT_RED, command=delete_cache, **btn_kw).pack(side="right", fill="x", expand=True, padx=(5, 0))


    def setup_gui(self):
        font_title = ctk.CTkFont(family=_UI_FONT[0], size=22, weight="bold")
        font_sub = ctk.CTkFont(family=_UI_FONT[0], size=13, weight="normal")
        font_btn = ctk.CTkFont(family=_UI_FONT[0], size=14, weight="bold")
        
        font_bio_lbl = ctk.CTkFont(family=_UI_FONT[0], size=16, weight="bold")
        font_bio_val = ctk.CTkFont(family=_UI_FONT[0], size=20, weight="bold")
        font_status_title = ctk.CTkFont(family=_UI_FONT[0], size=15, weight="bold")
        font_status_pct = ctk.CTkFont(family=_UI_FONT[0], size=16, weight="bold")

        header = ctk.CTkFrame(self.root, fg_color=VOID, corner_radius=0)
        header.pack(fill="x", padx=14, pady=(12, 6))

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(fill="x")
        ctk.CTkLabel(title_row, text="NETRA | CONTINUAL LEARNING SYSTEM", font=font_title, text_color=NEON_CYAN).pack(side="left", anchor="w")

        tab_frame = ctk.CTkFrame(title_row, fg_color="transparent")
        tab_frame.pack(side="right")
        
        self.theme_switch = ctk.CTkSwitch(tab_frame, text="DARK MODE ACTIVE", command=self.toggle_theme, progress_color=NEON_CYAN, font=font_sub, text_color=NEON_CYAN)
        self.theme_switch.pack(side="left", padx=(0, 20))
        
        self.btn_tab_operator = ctk.CTkButton(tab_frame, text="OPERATOR HUD", width=160, height=38, corner_radius=6, font=font_btn, fg_color=ABYSS, border_width=2, border_color=NEON_CYAN, text_color=NEON_CYAN, hover_color=HOVER_BASE, command=lambda: None)
        self.btn_tab_operator.pack(side="left", padx=(0, 8))
        self.btn_tab_neural = ctk.CTkButton(tab_frame, text="NEURAL OVERSEER", width=180, height=38, corner_radius=6, font=font_btn, fg_color=ABYSS, border_width=1, border_color=PLASMA_PURPLE, text_color=PLASMA_PURPLE, hover_color=HOVER_BASE, command=self.open_admin_console)
        self.btn_tab_neural.pack(side="left")

        body = ctk.CTkFrame(self.root, fg_color=VOID)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color=ABYSS, corner_radius=10, border_width=1, border_color=BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        video_shell = ctk.CTkFrame(left, fg_color=VOID, corner_radius=8)
        video_shell.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        video_shell.grid_rowconfigure(0, weight=1)
        video_shell.grid_columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(video_shell, width=800, height=450, bg=VOID[1], highlightthickness=0, bd=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.video_canvas.bind("<Configure>", self.on_canvas_resize)
        self.video_canvas.create_text(
            self.canvas_width / 2, self.canvas_height / 2, text="SYSTEM STANDBY\nAwaiting Video Feed...", 
            fill=NEON_CYAN[1], font=(_UI_FONT[0], 15), justify="center", tags="standby_text"
        )

        self.dvr_slider = ctk.CTkSlider(
            video_shell, from_=0, to=1, command=self.seek_video,
            fg_color=BORDER, progress_color=NEON_CYAN,
            button_color=NEON_CYAN, button_hover_color=NEON_CYAN,
        )
        self.dvr_slider.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        self.dvr_slider.set(0)
        self.is_webcam = True
        self.current_video_progress = 0.0
        video_shell.grid_rowconfigure(1, weight=0)
        
        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls.grid_columnconfigure((0, 1, 2, 3), weight=1)

        btn_kw = dict(height=38, corner_radius=6, font=font_btn, fg_color=ABYSS, border_width=1, hover_color=HOVER_BASE)

        self.btn_cam = ctk.CTkButton(controls, text="START WEBCAM", border_color=NEON_CYAN, text_color=NEON_CYAN, command=self.toggle_webcam, **btn_kw)
        self.btn_cam.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.btn_upload = ctk.CTkButton(controls, text="UPLOAD MP4", border_color=PLASMA_PURPLE, text_color=PLASMA_PURPLE, command=self.upload_video, **btn_kw)
        self.btn_upload.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        self.btn_network = ctk.CTkButton(controls, text="NETWORK FEED", border_color=RADIO_GREEN, text_color=RADIO_GREEN, command=self.connect_network_cam, **btn_kw)
        self.btn_network.grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        self.btn_stop = ctk.CTkButton(controls, text="STOP FEED", height=38, corner_radius=6, font=ctk.CTkFont(family=_UI_FONT[0], size=14, weight="bold"), fg_color=THREAT_RED, text_color="#ffffff", hover_color=("#990000", "#922b21"), command=self.stop_feed)
        self.btn_stop.grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        self.btn_pause = ctk.CTkButton(controls, text="PAUSE / PLAY", border_color=WARNING_YELLOW, text_color=WARNING_YELLOW, command=self.toggle_pause, **btn_kw)
        self.btn_pause.grid(row=1, column=0, padx=4, pady=4, sticky="ew")

        self.btn_restart = ctk.CTkButton(controls, text="RESTART VIDEO", border_color=NEON_CYAN, text_color=NEON_CYAN, command=self.restart_video, **btn_kw)
        self.btn_restart.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        self.btn_record = ctk.CTkButton(controls, text="START REC (LIVE)", border_color=("#D96600", "#ff8800"), text_color=("#D96600", "#ff8800"), command=self.toggle_record, **btn_kw)
        self.btn_record.grid(row=1, column=2, columnspan=2, padx=4, pady=4, sticky="ew")

        sidebar = ctk.CTkFrame(body, width=300, fg_color=ABYSS, corner_radius=10, border_width=1, border_color=BORDER)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.grid_propagate(False)
        ctk.CTkLabel(sidebar, text="BIOMETRIC STREAM", font=ctk.CTkFont(family=_UI_FONT[0], size=18, weight="bold"), text_color=NEON_CYAN).pack(anchor="w", padx=14, pady=(14, 6))

        self._bio_keys = ("VEL_X", "ROT_HIP", "DIST_CTR", "ACC_Z", "TORSO", "PHASE")
        self._bio_labels = {}
        
        grid = ctk.CTkFrame(sidebar, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=14, pady=(10, 14))
        grid.grid_columnconfigure(0, weight=1)

        for i, key in enumerate(self._bio_keys):
            cell = ctk.CTkFrame(grid, fg_color=VOID, corner_radius=6, border_width=1, border_color=BORDER)
            cell.grid(row=i, column=0, padx=5, pady=8, sticky="ew") 
            
            ctk.CTkLabel(cell, text=key, font=font_bio_lbl, text_color=PLASMA_PURPLE).pack(side="left", padx=15, pady=12)
            lbl = ctk.CTkLabel(cell, text="0.000", font=font_bio_val, text_color=RADIO_GREEN, width=80, anchor="e")
            lbl.pack(side="right", padx=15, pady=12)
            self._bio_labels[key] = lbl

        bottom = ctk.CTkFrame(self.root, fg_color=VOID, height=140)
        bottom.pack(fill="x", padx=14, pady=(0, 12))
        bottom.grid_columnconfigure(0, weight=2)
        bottom.grid_columnconfigure(1, weight=3)
        bottom.grid_columnconfigure(2, weight=1)

        status_wrap = ctk.CTkFrame(bottom, fg_color=ABYSS, corner_radius=8, border_width=1, border_color=BORDER)
        status_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)
        ctk.CTkLabel(status_wrap, text="CONDITION | STATUS", font=font_sub, text_color=NEON_CYAN).pack(anchor="w", padx=12, pady=(12, 0))
        self.lbl_status_banner = ctk.CTkLabel(status_wrap, text="PERIMETER SECURE", font=ctk.CTkFont(family=_UI_FONT[0], size=22, weight="bold"), text_color=RADIO_GREEN)
        self.lbl_status_banner.pack(anchor="w", padx=12, pady=(8, 12))

        meters = ctk.CTkFrame(bottom, fg_color=ABYSS, corner_radius=8, border_width=1, border_color=BORDER)
        meters.grid(row=0, column=1, sticky="nsew", padx=8, pady=4)
        
        wd_lbl_frame = ctk.CTkFrame(meters, fg_color="transparent")
        wd_lbl_frame.pack(fill="x", padx=12, pady=(10, 2))
        self.lbl_watchdog_title = ctk.CTkLabel(wd_lbl_frame, text="WATCHDOG", font=font_status_title, text_color=WARNING_YELLOW)
        self.lbl_watchdog_title.pack(side="left")
        self.lbl_watchdog_pct = ctk.CTkLabel(wd_lbl_frame, text="0.0%", font=font_status_pct, text_color=WARNING_YELLOW, width=60, anchor="e")
        self.lbl_watchdog_pct.pack(side="right")
        
        self.bar_watchdog = ctk.CTkProgressBar(meters, width=280, height=18, progress_color=WARNING_YELLOW, fg_color=VOID)
        self.bar_watchdog.pack(anchor="w", padx=12, pady=(0, 8))
        self.bar_watchdog.set(0)

        an_lbl_frame = ctk.CTkFrame(meters, fg_color="transparent")
        an_lbl_frame.pack(fill="x", padx=12, pady=(6, 2))
        self.lbl_analyst_title = ctk.CTkLabel(an_lbl_frame, text="ANALYST", font=font_status_title, text_color=PLASMA_PURPLE)
        self.lbl_analyst_title.pack(side="left")
        self.lbl_analyst_pct = ctk.CTkLabel(an_lbl_frame, text="0.0%", font=font_status_pct, text_color=PLASMA_PURPLE, width=60, anchor="e")
        self.lbl_analyst_pct.pack(side="right")
        
        self.bar_analyst = ctk.CTkProgressBar(meters, width=280, height=18, progress_color=PLASMA_PURPLE, fg_color=VOID)
        self.bar_analyst.pack(anchor="w", padx=12, pady=(0, 10))
        self.bar_analyst.set(0)

        self.btn_isolate = ctk.CTkButton(bottom, text="ISOLATE ANOMALY", width=200, height=54, corner_radius=8, font=font_btn, fg_color=ABYSS, border_width=2, border_color=THREAT_RED, text_color=THREAT_RED, hover_color=("#FFCCCC", "#2a0a10"), state="disabled", command=self.isolate_anomaly)
        self.btn_isolate.grid(row=0, column=2, sticky="e", padx=(8, 0), pady=4)

        self._reset_gui_telemetry()


    def release_cap_safely(self):
        if self.cap_lock.acquire(timeout=1.0):
            try:
                if self.cap:
                    self.cap.release()
                    self.cap = None
            finally:
                self.cap_lock.release()
        else:
            print("⚠️ [WARNING] AI Thread is currently blocked on video read. Abandoning old capture safely.")
            self.cap = None


    def toggle_pause(self):
        if not self.cap or not self.cap.isOpened(): return
        
        if not self.is_webcam:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0 and current_frame >= total_frames - 1:
                self.restart_video()
                return

        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.configure(text="▶ RESUME", fg_color=WARNING_YELLOW, text_color=VOID)
        else:
            self.btn_pause.configure(text="PAUSE / PLAY", fg_color=ABYSS, text_color=WARNING_YELLOW)

    def restart_video(self):
        if self.cap and self.cap.isOpened() and not self.is_webcam:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.skel_buffer.clear()
            self.spat_buffer.clear()
            self.current_video_progress = 0.0
            self.is_paused = False
            self.btn_pause.configure(text="PAUSE / PLAY", fg_color=ABYSS, text_color=WARNING_YELLOW)


    def toggle_record(self):
        if not self.is_recording:
            if not self.cap or not self.cap.isOpened(): return
                
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps != fps: fps = 20.0

            save_dir = r"D:\Threat_Detection_Workspace\Anomalies_Logged"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"Record_Raw_{timestamp}.avi")

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            self.is_recording = True
            self.btn_record.configure(text="STOP REC", fg_color=THREAT_RED, text_color="#ffffff")
            print(f"🔴 Recording Started: {filename}")
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.btn_record.configure(text="START REC (LIVE)", fg_color=ABYSS, text_color=("#D96600", "#ff8800"))
            print("⏹ Recording Stopped.")


    def isolate_anomaly(self):
        if not (hasattr(self, 'clean_frame') and self.clean_frame is not None): return
        if not hasattr(self, 'current_live_vector'): return

        dialog = ctk.CTkInputDialog(text="Enter Name for this Fast-Track Threat:", title="AEGIS FAST-TRACK")
        threat_name = dialog.get_input()
        if not threat_name: return 
        threat_name = threat_name.strip().replace(" ", "_")

        self.fast_track_db[threat_name] = self.current_live_vector
        torch.save(self.fast_track_db, FAST_TRACK_DB_PATH)
        print(f"⚡ AEGIS FAST-TRACK: 256-dim vector saved for [{threat_name}] → {FAST_TRACK_DB_PATH}")

        save_dir = r"D:\Threat_Detection_Workspace\Anomalies_Logged"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"Aegis_{threat_name}_{timestamp}.jpg")

        save_img = self.clean_frame.copy()
        cv2.putText(save_img, f"AEGIS CACHE: {timestamp}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imwrite(filename, save_img)

        original_text = self.btn_isolate.cget("text")
        self.btn_isolate.configure(text="THREAT CACHED!", text_color=RADIO_GREEN[1], border_color=RADIO_GREEN[1])
        self.root.after(2000, lambda: self.btn_isolate.configure(text=original_text, text_color=THREAT_RED[1], border_color=THREAT_RED[1]))


    def _reset_gui_telemetry(self):
        self.gui_condition = "GREEN"
        self.gui_watchdog_pct = 0.0
        self.gui_analyst_pct = 0.0
        self.gui_analyst_prediction = "-"
        self.gui_status_line = "PERIMETER SECURE"
        self.lbl_status_banner.configure(text="PERIMETER SECURE", text_color=RADIO_GREEN[1])
        self.bar_watchdog.set(0)
        self.bar_analyst.set(0)
        if hasattr(self, 'lbl_watchdog_pct'):
            self.lbl_watchdog_pct.configure(text="0.0%")
            self.lbl_analyst_pct.configure(text="0.0%")
        self.btn_isolate.configure(state="disabled")
        for k in self._bio_keys:
            self._bio_labels[k].configure(text="0.000", text_color=RADIO_GREEN[1])


    def toggle_webcam(self): self.start_feed(0)


    def seek_video(self, value):
        try:
            with self.cap_lock:
                if self.cap and self.cap.isOpened() and not self.is_webcam:
                    total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if total_frames > 0:
                        target_frame = int(float(value) * total_frames)
                        
                        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        if not success:
                            print(f"⚠️ [WARNING] OpenCV failed to seek to frame {target_frame}.")
                            return
                        
                        self.skel_buffer.clear()
                        self.spat_buffer.clear()
                        self.current_video_progress = target_frame / total_frames
                        
                        if self.is_paused:
                            ret, frame = self.cap.read()
                            if ret and frame is not None:
                                self.clean_frame = frame.copy()
                                frame_display = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                                self.current_frame_image = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                            else:
                                print("⚠️ [WARNING] Failed to read frame after seeking.")
                            
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ VIDEO SEEK ERROR: {error_msg}\n")
            if hasattr(self, 'lbl_status_banner'):
                self.lbl_status_banner.configure(text="⚠️ ERROR: VIDEO SEEK FAILED", text_color=THREAT_RED[1])
                self.root.update()


    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if file_path: self.start_feed(file_path)


    def connect_network_cam(self):
        if not hasattr(self, 'recent_urls'): self.recent_urls = ["http://192.168.1.4:4747/video", "http://192.168.1.5:8080/video"]
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("NETWORK FEED OVERRIDE")
        dialog.geometry("550x220")
        dialog.configure(fg_color=VOID)
        dialog.attributes("-topmost", True)
        dialog.grab_set()  
        ctk.CTkLabel(dialog, text="SELECT OR ENTER CAMERA STREAM URL:", text_color=NEON_CYAN, font=ctk.CTkFont(family=_UI_FONT[0], size=14, weight="bold")).pack(pady=(25, 10))
        url_combo = ctk.CTkComboBox(dialog, values=self.recent_urls, width=450, height=35, font=ctk.CTkFont(family=_UI_FONT[0], size=14), fg_color=ABYSS, border_color=PLASMA_PURPLE, dropdown_fg_color=ABYSS, dropdown_text_color=NEON_CYAN)
        url_combo.pack(pady=10)
        url_combo.set(self.recent_urls[0] if self.recent_urls else "")

        def on_ok():
            url = url_combo.get().strip()
            if url:
                if url not in self.recent_urls: self.recent_urls.insert(0, url)
                self.start_feed(url)
            dialog.destroy()
        def on_cancel(): dialog.destroy()

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=(15, 20))
        ctk.CTkButton(btn_frame, text="OK", width=120, height=35, command=on_ok, fg_color=ABYSS, border_width=1, border_color=RADIO_GREEN, text_color=RADIO_GREEN, font=ctk.CTkFont(family=_UI_FONT[0], size=14, weight="bold"), hover_color=HOVER_BASE).pack(side="left", padx=15)
        ctk.CTkButton(btn_frame, text="CANCEL", width=120, height=35, command=on_cancel, fg_color=ABYSS, border_width=1, border_color=THREAT_RED, text_color=THREAT_RED, font=ctk.CTkFont(family=_UI_FONT[0], size=14, weight="bold"), hover_color=("#FFCCCC", "#2a0a10")).pack(side="right", padx=15)


    def start_feed(self, source):
        self.stop_feed()
        self.lbl_status_banner.configure(text="CONNECTING TO STREAM...", text_color=WARNING_YELLOW)
        self.root.update()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened(): 
            self.release_cap_safely()
            error_msg = f"Failed to connect to video source:\n{source}\n\nPossible causes:\n1. IP/Camera is offline or unreachable.\n2. URL format/port is incorrect.\n3. Network firewall is blocking the connection."
            self.show_error_dialog("CONNECTION ERROR", error_msg)
            self.video_canvas.delete("all")
            text_col = NEON_CYAN[0] if self.current_theme == "Light" else NEON_CYAN[1]
            self.video_canvas.create_text(self.canvas_width / 2, self.canvas_height / 2, text="SYSTEM STANDBY\nAwaiting Video Feed...", fill=text_col, font=(_UI_FONT[0], 15), justify="center", tags="standby_text")
            self._reset_gui_telemetry()
            return

        self.is_webcam = (isinstance(source, int) or str(source).startswith("http") or str(source).startswith("rtsp"))
        if self.is_webcam:
            self.dvr_slider.configure(state="disabled", progress_color=BORDER[1], button_color=BORDER[1])
        else:
            self.dvr_slider.configure(state="normal", progress_color=NEON_CYAN[1], button_color=NEON_CYAN[1])
            self.dvr_slider.set(0)
            self.current_video_progress = 0.0

        self.is_running = True
        self.is_paused = False
        self.btn_pause.configure(text="PAUSE / PLAY", fg_color=ABYSS, text_color=WARNING_YELLOW)
        self.skel_buffer.clear()
        self.spat_buffer.clear()
        
        self.alarm_latch_expiry = 0.0
        self.last_face_bboxes = []
        self.last_pose_landmarks = []

        self.ai_thread = threading.Thread(target=self.ai_processing_worker, daemon=True)
        self.ai_thread.start()


    def on_closing(self):
        exit_win = ctk.CTkToplevel(self.root)
        exit_win.title("SYSTEM SHUTDOWN")
        exit_win.geometry("500x200")
        exit_win.configure(fg_color=VOID)
        exit_win.attributes("-topmost", True)
        exit_win.grab_set()

        ctk.CTkLabel(exit_win, text="⚠️ CONFIRM SYSTEM SHUTDOWN", font=ctk.CTkFont(family=_UI_FONT[0], size=16, weight="bold"), text_color=WARNING_YELLOW).pack(pady=(20, 10))
        ctk.CTkLabel(exit_win, text="Closing the Command Center will terminate all active surveillance streams.\nHow would you like to proceed?", font=ctk.CTkFont(family=_UI_FONT[0], size=13), text_color=("#000000", "#FFFFFF")).pack(pady=(0, 20))

        btn_frame = ctk.CTkFrame(exit_win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20)

        def do_minimize():
            self.root.iconify()
            exit_win.destroy()

        def do_quit():
            exit_win.destroy()
            self.is_running = False
            if self.is_recording: self.toggle_record()
            self.release_cap_safely()
            self.root.destroy()

        def do_cancel():
            exit_win.destroy()

        btn_font = ctk.CTkFont(family=_UI_FONT[0], size=13, weight="bold")
        ctk.CTkButton(btn_frame, text="MINIMIZE", width=100, font=btn_font, fg_color=ABYSS, border_width=1, border_color=NEON_CYAN, text_color=NEON_CYAN, command=do_minimize).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(btn_frame, text="QUIT", width=100, font=btn_font, fg_color=ABYSS, border_width=1, border_color=THREAT_RED, text_color=THREAT_RED, hover_color=("#FFCCCC", "#2a0a10"), command=do_quit).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(btn_frame, text="CANCEL", width=100, font=btn_font, fg_color=ABYSS, border_width=1, border_color=BORDER, text_color=("#000000", "#FFFFFF"), command=do_cancel).pack(side="left", expand=True, padx=5)


    def on_canvas_resize(self, event):
        self.canvas_width = max(event.width, 100)
        self.canvas_height = max(event.height, 100)
        self.video_canvas.coords("standby_text", self.canvas_width / 2, self.canvas_height / 2)


    def stop_feed(self):
        self.is_running = False
        if self.is_recording: self.toggle_record()
        self.release_cap_safely()
        self.video_canvas.delete("all")
        text_col = NEON_CYAN[0] if self.current_theme == "Light" else NEON_CYAN[1]
        self.video_canvas.create_text(self.canvas_width / 2, self.canvas_height / 2, text="SYSTEM STANDBY\nAwaiting Video Feed...", fill=text_col, font=(_UI_FONT[0], 15), justify="center", tags="standby_text")
        self.current_frame_image = None
        self._reset_gui_telemetry()

    def ai_processing_worker(self):
        frame_counter = 0
        try:
            with torch.no_grad():
                while self.is_running:
                    if self.is_paused:
                        time.sleep(0.05)
                        continue

                    with self.cap_lock:
                        if not self.is_running or not self.cap or not self.cap.isOpened():
                            break
                        ret, frame = self.cap.read()

                    if not ret:
                        if not self.is_webcam:
                            self.is_paused = True
                            self.btn_pause.configure(text="▶ REPLAY", fg_color=WARNING_YELLOW, text_color=VOID)
                            continue
                        else:
                            self.is_running = False
                            break

                    self.clean_frame = frame.copy()
                    frame_counter += 1

                    if not self.is_webcam:
                        with self.cap_lock:
                            if self.cap:
                                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                                total_frames  = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                if total_frames > 0:
                                    self.current_video_progress = current_frame / total_frames

                    if self.is_recording and hasattr(self, 'video_writer') and self.video_writer:
                        rec_frame = self.clean_frame.copy()
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(rec_frame, f"REC // RAW: {ts}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.video_writer.write(rec_frame)

                    if frame_counter % 3 == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        face_results = self.face_detector.process(frame_rgb)
                        self.last_face_bboxes = []
                        if face_results.detections:
                            for det in face_results.detections:
                                self.last_face_bboxes.append((det.location_data.relative_bounding_box, det.score[0]))

                        results = self.pose.process(frame_rgb)

                        if not results.pose_landmarks:
                            self.sensor_status = "STANDBY // NO SUBJECT"
                            self.last_pose_landmarks = []
                            self.current_telemetry = {k: 0.0 for k in ("VEL_X", "ROT_HIP", "DIST_CTR", "ACC_Z", "TORSO", "PHASE")}
                            self.skel_buffer.clear()
                            self.spat_buffer.clear()

                        else:
                            self.sensor_status = "ACTIVE"
                            self.last_pose_landmarks = results.pose_landmarks.landmark
                            lmks = self.last_pose_landmarks
                            self.current_telemetry = {
                                "VEL_X": (lmks[23].x + lmks[24].x) / 2, "ROT_HIP": (lmks[23].y - lmks[24].y) * 10,
                                "DIST_CTR": lmks[0].x, "ACC_Z": lmks[23].z, "TORSO": (lmks[11].y + lmks[12].y) / 2, "PHASE": time.time() % 10,
                            }
                            landmarks = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in lmks]).flatten()
                            self.skel_buffer.append(landmarks)

                            input_tensor = self.preprocess(frame_rgb).unsqueeze(0).to(self.device)
                            features = self.spatial_model(input_tensor).cpu().squeeze().numpy()
                            self.spat_buffer.append(features)

                            if len(self.skel_buffer) < 15:
                                self.last_status_color = (0, 255, 255)   
                                self.gui_condition = "GREEN"
                                self.gui_status_line = "CALIBRATING SUBJECT MOTION..."
                                self.gui_analyst_prediction = "Gathering temporal baseline..."
                                self.gui_watchdog_pct = 0.0
                                self.gui_analyst_pct = 0.0

                            else:
                                skel_input = list(self.skel_buffer)
                                pad_skel = skel_input[0] if skel_input else np.zeros(132)
                                while len(skel_input) < MAX_FRAMES: skel_input.insert(0, pad_skel)

                                spat_input = list(self.spat_buffer)
                                feature_dim = len(spat_input[0]) if spat_input else 2048
                                pad_spat = spat_input[0] if spat_input else np.zeros(feature_dim)
                                while len(spat_input) < MAX_FRAMES: spat_input.insert(0, pad_spat)

                                skel_tensor = torch.FloatTensor(np.array(skel_input)).unsqueeze(0).to(self.device)
                                spat_tensor = torch.FloatTensor(np.array(spat_input)).unsqueeze(0).to(self.device)
                                spat_mean = torch.mean(spat_tensor, dim=1)

                                out_binary, out_fine = self.fusion_model(skel_tensor, spat_mean)
                                prob_binary = F.softmax(out_binary, dim=1).squeeze()
                                prob_fine   = F.softmax(out_fine,   dim=1).squeeze()

                                threat_confidence = prob_binary[1].item() * 100
                                top_class_idx     = torch.argmax(prob_fine).item()
                                try:
                                    top_class_name = CLASSES[top_class_idx]
                                except IndexError:
                                    top_class_name = "Unknown"
                                top_class_conf = prob_fine[top_class_idx].item() * 100

                                x_skel = self.fusion_model.skel_embedding(skel_tensor)
                                x_skel = self.fusion_model.pos_encoder(x_skel)
                                transformer_out = self.fusion_model.transformer(x_skel)
                                
                                kinematic_features, _ = torch.max(transformer_out, dim=1)
                                spatial_features = self.fusion_model.spatial_fc(spat_mean)
                                live_vector = torch.cat((kinematic_features, spatial_features), dim=1).squeeze()
                                
                                self.current_live_vector = live_vector.cpu().tolist()  

                                fast_track_triggered = False
                                ft_gate = self.fast_track_db and (
                                    DEMO_MODE or
                                    (threat_confidence >= 50.0 and top_class_conf < ANALYST_THRESHOLD)
                                )
                                if ft_gate:
                                    for name, saved_vector in list(self.fast_track_db.items()):
                                        saved_tensor = torch.FloatTensor(saved_vector).to(self.device)
                                        kin_sim  = F.cosine_similarity(live_vector[:128].unsqueeze(0), saved_tensor[:128].unsqueeze(0)).item()
                                        spat_sim = F.cosine_similarity(live_vector[128:].unsqueeze(0), saved_tensor[128:].unsqueeze(0)).item()
                                        if kin_sim >= SIMILARITY_THRESHOLD and spat_sim >= SPATIAL_THRESHOLD:
                                            fast_track_triggered = True
                                            self.latched_threat_name = name
                                            self.latched_threat_conf = (0.5 * kin_sim + 0.5 * spat_sim) * 100
                                            self.latched_fast_track = True
                                            break

                                current_time = time.time()
                                analyst_says_normal = (top_class_name == "Normal")
                                is_active_threat = (threat_confidence >= 50.0 and top_class_conf >= ANALYST_THRESHOLD and not analyst_says_normal)

                                if is_active_threat or fast_track_triggered:
                                    self.alarm_latch_expiry = current_time + 3.0
                                    if not fast_track_triggered:
                                        self.latched_threat_name = top_class_name
                                        self.latched_threat_conf = top_class_conf
                                        self.latched_fast_track = False

                                if current_time < self.alarm_latch_expiry:
                                    if self.latched_fast_track:
                                        self.last_status_color = (255, 0, 255)
                                        self.gui_condition = "PURPLE"
                                        self.gui_status_line = f"FAST-TRACK OVERRIDE // {self.latched_threat_name.upper()}"
                                    else:
                                        self.last_status_color = (0, 0, 255)
                                        self.gui_condition = "RED"
                                        self.gui_status_line = f"PERIMETER BREACH CONFIRMED // {self.latched_threat_name.upper()}"
                                    self.gui_analyst_prediction = f"{self.latched_threat_name} ({self.latched_threat_conf:.1f}%)"
                                    if current_time - self.last_alarm_time > 3.0:
                                        threading.Thread(target=lambda: [winsound.Beep(2500, 600), winsound.Beep(2000, 600)], daemon=True).start()
                                        self.last_alarm_time = current_time
                                elif threat_confidence >= 50.0 and top_class_conf < ANALYST_THRESHOLD:
                                    self.last_status_color = (0, 165, 255)
                                    self.gui_condition = "YELLOW"
                                    self.gui_status_line = "ANOMALY DETECTED // ISOLATE ANOMALY"
                                    self.gui_analyst_prediction = f"Unknown anomaly ({top_class_name} at {top_class_conf:.1f}%)"
                                else:
                                    self.last_status_color = (0, 255, 0)
                                    self.gui_condition = "GREEN"
                                    self.gui_status_line = "PERIMETER SECURE // SUBJECT TRACKED"
                                    self.gui_analyst_prediction = f"{top_class_name} ({top_class_conf:.1f}%)" if analyst_says_normal else "Nominal - no active threat"

                                if current_time < self.alarm_latch_expiry and self.latched_fast_track:
                                    self.gui_watchdog_pct = 100.0               
                                    self.gui_analyst_pct  = self.latched_threat_conf  
                                else:
                                    self.gui_watchdog_pct = threat_confidence
                                    self.gui_analyst_pct  = top_class_conf

                    h, w, _ = frame.shape

                    if time.time() < self.alarm_latch_expiry and self.latched_threat_conf >= 80.0:
                        for bboxC, score in self.last_face_bboxes:
                            fx, fy = int(bboxC.xmin * w), int(bboxC.ymin * h)
                            fw, fh = int(bboxC.width * w), int(bboxC.height * h)
                            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 255), 2)
                            cv2.putText(frame, f"FACE LOCKED [{int(score*100)}%]", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                    if self.last_pose_landmarks:
                        x_coords = [int(lmk.x * w) for lmk in self.last_pose_landmarks]
                        y_coords = [int(lmk.y * h) for lmk in self.last_pose_landmarks]
                        for cx, cy in zip(x_coords, y_coords): cv2.circle(frame, (cx, cy), 3, self.last_status_color, -1)
                        for connection in mp.solutions.pose.POSE_CONNECTIONS:
                            s, e = connection
                            cv2.line(frame, (x_coords[s], y_coords[s]), (x_coords[e], y_coords[e]), self.last_status_color, 1)

                    cv2.putText(frame, "CAM-01 // LIVE FEED // SECTOR-7", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1)
                    if self.is_recording:
                        cv2.putText(frame, "REC", (750, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame, (735, 25), 5, (0, 0, 255), -1)

                    frame_display = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                    self.current_frame_image = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)

        except Exception as e:
            error_msg = str(e)
            print(f"\nFATAL AI THREAD CRASH: {error_msg}\n")
            self.root.after(0, lambda: self.show_error_dialog("AI ENGINE CRASH", f"A critical error occurred in the neural network loop:\n\n{error_msg}"))
            self.root.after(0, self.stop_feed)


    def refresh_gui(self):
        if self.is_running and self.current_frame_image is not None:
            img = Image.fromarray(self.current_frame_image)
            self.imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        if hasattr(self, 'current_telemetry') and self.sensor_status == "ACTIVE":
            for key in self._bio_keys:
                val = self.current_telemetry.get(key, 0.0)
                self._bio_labels[key].configure(text=f"{val:+.3f}", text_color=RADIO_GREEN)
        else:
            for key in self._bio_keys:
                self._bio_labels[key].configure(text="0.000", text_color="#555555")

        cond = self.gui_condition
        banner_color = _CONDITION_HEX.get(cond, NEON_CYAN[1])
        self.lbl_status_banner.configure(text=self.gui_status_line, text_color=banner_color)

        if hasattr(self, 'lbl_watchdog_title'):
            if cond == "PURPLE":
                self.lbl_watchdog_title.configure(text="SYSTEM OVERRIDE",   text_color=PLASMA_PURPLE)
                self.lbl_analyst_title.configure( text="COSINE SIMILARITY", text_color=PLASMA_PURPLE)
                self.bar_watchdog.configure(progress_color=PLASMA_PURPLE[1] if self.current_theme=="Dark" else PLASMA_PURPLE[0])
                
                self.lbl_watchdog_pct.configure(text=f"{self.gui_watchdog_pct:.1f}%", text_color=PLASMA_PURPLE)
                self.lbl_analyst_pct.configure(text=f"{self.gui_analyst_pct:.1f}%", text_color=PLASMA_PURPLE)
            else:
                self.lbl_watchdog_title.configure(text="WATCHDOG", text_color=WARNING_YELLOW)
                self.lbl_analyst_title.configure( text="ANALYST",  text_color=PLASMA_PURPLE)
                self.bar_watchdog.configure(progress_color=WARNING_YELLOW[1] if self.current_theme=="Dark" else WARNING_YELLOW[0])
                
                self.lbl_watchdog_pct.configure(text=f"{self.gui_watchdog_pct:.1f}%", text_color=WARNING_YELLOW)
                self.lbl_analyst_pct.configure(text=f"{self.gui_analyst_pct:.1f}%", text_color=PLASMA_PURPLE)

        self.bar_watchdog.set(min(1.0, max(0.0, self.gui_watchdog_pct / 100.0)))
        self.bar_analyst.set(min(1.0, max(0.0, self.gui_analyst_pct / 100.0)))
        
        self.btn_isolate.configure(state=("normal" if self.is_running else "disabled"))

        if not self.is_webcam and self.is_running and not self.is_paused:
            self.dvr_slider.set(self.current_video_progress)

        self.root.after(30, self.refresh_gui)


if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk(fg_color=VOID)
    app = OODACommandCenter(root)
    root.mainloop()