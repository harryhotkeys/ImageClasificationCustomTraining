# ═══════════════════════════════════════════════════════════════
# STEP 1 — Imports and Configuration
# ═══════════════════════════════════════════════════════════════

"""
Custom Image Classifier
Requirements: pip install torch torchvision pillow
"""

import os, json, shutil, threading, time
from pathlib import Path
from datetime import timedelta

import torch, torch.nn as nn, torch.optim as optim
from torchvision import models, transforms
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, simpledialog, scrolledtext, ttk

# Configuration
IMAGES_FOLDER = "images"
CUSTOM_DATA_DIR = "custom_classes"
MODEL_PATH = "custom_head.pth"
META_PATH = "custom_classes.json"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
INPUT_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EPOCHS = 30
LR = 0.01


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Image Preprocessing Functions
# ═══════════════════════════════════════════════════════════════

def get_transform():
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def preprocess_image(path, transform=None):
    transform = transform or get_transform()
    img = Image.open(path).convert("RGB")
    return img, transform(img).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
# STEP 3 — The CustomClassifier Class
# ═══════════════════════════════════════════════════════════════

class CustomClassifier:

    def __init__(self):
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = None
        self.classes = {}
        self.trained = False
        self._load()

    def _load(self):
        if not os.path.exists(META_PATH):
            return
        with open(META_PATH) as f:
            self.classes = json.load(f)
        n = len(self.classes)
        if n > 0:
            self.head = nn.Linear(self.feature_dim, n)
            if os.path.exists(MODEL_PATH):
                self.head.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
                self.trained = True
            self.head.eval()

    def save(self):
        with open(META_PATH, "w") as f:
            json.dump(self.classes, f, indent=2)
        if self.head:
            torch.save(self.head.state_dict(), MODEL_PATH)

    def add_class(self, name):
        if name not in self.classes:
            self.classes[name] = len(self.classes)
            self.head = nn.Linear(self.feature_dim, len(self.classes))
            self.trained = False

    def remove_class(self, name):
        del self.classes[name]
        self.classes = {n: i for i, n in enumerate(self.classes)}
        self.head = nn.Linear(self.feature_dim, len(self.classes)) if self.classes else None
        self.trained = False

    def features(self, tensor):
        with torch.no_grad():
            return self.backbone(tensor)

    def predict(self, tensor):
        pass  # Step 12

    def train_model(self, log=None, progress=None):
        pass  # Step 11


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Helper + GUI Window and Styling
# ═══════════════════════════════════════════════════════════════

def format_duration(s):
    if s is None: return "--:--"
    return str(timedelta(seconds=int(round(s)))).split(", ")[-1]


class ModernClassifierApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Custom Image Classifier")
        self.geometry("1080x800")
        self.minsize(960, 640)

        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.primary = "#0EA5A4"
        self.accent = "#7C3AED"
        self.surface = "#0f1724"
        self.card = "#0b1220"
        self.text = "#E6EEF8"
        self.muted = "#9AA6B2"
        self.font_heading = ("Inter", 16, "bold")
        self.font_normal = ("Inter", 11)
        self.font_mono = ("Consolas", 10)
        self.font_small = ("Inter", 9)

        self._setup_styles()

        self.model = None
        self.current_photo = None
        self.selected_folder = IMAGES_FOLDER

        self._build_ui()
        self._set_status("Loading backbone...")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _setup_styles(self):
        s = self.style
        s.configure("TFrame", background=self.surface)
        s.configure("Card.TFrame", background=self.card, relief="flat")
        s.configure("TLabel", background=self.surface, foreground=self.text, font=self.font_normal)
        s.configure("Card.TLabel", background=self.card, foreground=self.text, font=self.font_normal)
        s.configure("Heading.TLabel", background=self.surface, foreground=self.text, font=self.font_heading)
        s.configure("Muted.TLabel", background=self.surface, foreground=self.muted, font=self.font_normal)
        s.configure("CardMuted.TLabel", background=self.card, foreground=self.muted, font=self.font_small)
        s.configure("Mono.TLabel", background=self.card, foreground=self.text, font=self.font_mono)
        s.configure("TButton", font=self.font_normal, padding=6)
        s.configure("Accent.TButton", foreground="#fff", background=self.accent)
        s.map("Accent.TButton", background=[("active", self.primary)])
        s.configure("Primary.TButton", foreground="#fff", background=self.primary)
        s.configure("Danger.TButton", foreground="#fff", background="#EF4444")
        s.configure("Success.TButton", foreground="#fff", background="#22C55E")
        s.configure("TProgressbar", troughcolor=self.card, background=self.primary)

    # ═══════════════════════════════════════════════════════════
    # STEP 5 — Left Panel (Preview + Activity Log)
    # STEP 6 — Right Panel (Classification Controls)
    # ═══════════════════════════════════════════════════════════

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=(18, 12))
        top.pack(fill=tk.X)
        ttk.Label(top, text="Custom Image Classifier", style="Heading.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, text="Train your own classes", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 0))

        main = ttk.Frame(self, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        # Status bar (bottom)
        self.status_var = tk.StringVar(value="Idle")
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(bar, textvariable=self.status_var, style="Muted.TLabel", padding=8).pack(side=tk.LEFT)

        # --- STEP 5: LEFT PANEL ---
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        preview = ttk.Frame(left, style="Card.TFrame", padding=12)
        preview.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(preview, text="Preview", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.preview_label = ttk.Label(preview, text="No image", anchor="center", style="Card.TLabel")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(10, 6))

        info = ttk.Frame(preview, style="Card.TFrame")
        info.pack(fill=tk.X, pady=(4, 4))
        self.info_text = tk.StringVar(value="Prediction: —")
        ttk.Label(info, textvariable=self.info_text, style="Mono.TLabel").pack(anchor=tk.W, padx=4, pady=6)

        log_card = ttk.Frame(left, style="Card.TFrame", padding=8)
        log_card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(log_card, text="Activity Log", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.log_box = scrolledtext.ScrolledText(
            log_card, height=14, bg=self.card, fg=self.text,
            insertbackground=self.text, wrap=tk.WORD, font=self.font_mono, relief=tk.FLAT)
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        self.log_box.insert(tk.END, "Ready. Add classes and train before classifying.\n")
        self.log_box.config(state=tk.DISABLED)

        # --- STEP 6: RIGHT PANEL ---
        right = ttk.Frame(main, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, anchor=tk.N)

        controls = ttk.Frame(right, style="Card.TFrame", padding=12)
        controls.pack(fill=tk.X)

        # Single image
        ttk.Label(controls, text="Classify Single Image", style="Mono.TLabel").pack(anchor=tk.W, pady=(0, 6))
        row1 = ttk.Frame(controls, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=(0, 6))
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(row1, textvariable=self.path_var, width=28)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.path_entry.bind("<Return>", lambda e: self._start_single())
        ttk.Button(row1, text="Browse", command=self._browse_file).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(row1, text="Classify", style="Accent.TButton",
                   command=self._start_single).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Folder
        ttk.Label(controls, text="Folder Classification", style="Mono.TLabel").pack(anchor=tk.W, pady=(0, 6))
        row2 = ttk.Frame(controls, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=(0, 4))
        self.folder_label_var = tk.StringVar(value=f"(default) {IMAGES_FOLDER}")
        ttk.Label(row2, textvariable=self.folder_label_var, style="Card.TLabel",
                  wraplength=200).pack(side=tk.LEFT)
        ttk.Button(row2, text="Choose...", command=self._choose_folder).pack(side=tk.RIGHT, padx=(6, 0))
        self.classify_folder_btn = ttk.Button(controls, text="Classify Folder",
                                              style="Primary.TButton", command=self._start_folder)
        self.classify_folder_btn.pack(fill=tk.X, pady=(6, 4))

        # Progress
        prog = ttk.Frame(controls, style="Card.TFrame", padding=6)
        prog.pack(fill=tk.X, pady=(6, 4))
        ttk.Label(prog, text="Progress", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.progress = ttk.Progressbar(prog, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, pady=(4, 4))
        prow = ttk.Frame(prog, style="Card.TFrame")
        prow.pack(fill=tk.X)
        self.percent_var = tk.StringVar(value="0%")
        self.eta_var = tk.StringVar(value="ETA: --:--")
        ttk.Label(prow, textvariable=self.percent_var, style="Card.TLabel", width=7).pack(side=tk.LEFT)
        ttk.Label(prow, textvariable=self.eta_var, style="Card.TLabel").pack(side=tk.LEFT)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(controls, text="Clear Log", command=self._clear_log).pack(fill=tk.X)

        # ═══════════════════════════════════════════════════════
        # STEP 7 — Training Panel (you will add this next)
        train_card = ttk.Frame(right, style="Card.TFrame", padding=12)
        train_card.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(train_card, text="Training", style="Mono.TLabel").pack(anchor=tk.W, pady=(0, 6))
        self.custom_list_var = tk.StringVar(value="(none)")
        ttk.Label(train_card, textvariable=self.custom_list_var, style="CardMuted.TLabel", wraplength=300).pack(
            anchor=tk.W, pady=(0, 6))
        btn_row = ttk.Frame(train_card, style="Card.TFrame")
        btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_row, text="+ Add Class", style="Success.TButton", command=self._add_class).pack(side=tk.LEFT,padx=(0, 4))
        ttk.Button(btn_row, text="Remove Class", style="Danger.TButton", command=self._remove_class).pack(side=tk.LEFT)
        ttk.Button(train_card, text="Train", style="Accent.TButton", command=self._start_training).pack(fill=tk.X,pady=(6, 4))
        self.train_status_var = tk.StringVar(value="Not trained")
        ttk.Label(train_card, textvariable=self.train_status_var, style="CardMuted.TLabel").pack(anchor=tk.W,pady=(2, 0))
        # ═══════════════════════════════════════════════════════

        # Training panel placeholder — needed so _refresh_ui doesn't crash
        self.custom_list_var = tk.StringVar(value="(none)")
        self.train_status_var = tk.StringVar(value="Not trained")

    # ═══════════════════════════════════════════════════════════
    # Stub methods — these make the app runnable without crashing.
    # You will replace each one in later steps.
    # ═══════════════════════════════════════════════════════════

    def _load_model_thread(self):
        self.model = CustomClassifier()
        self._set_status("Backbone loaded")
        self._log("ResNet-18 backbone loaded (feature extractor).")
        self._refresh_ui()

    def _log(self, text):
        def _do():
            self.log_box.config(state=tk.NORMAL)
            self.log_box.insert(tk.END, f"{text}\n")
            self.log_box.see(tk.END)
            self.log_box.config(state=tk.DISABLED)

        self.after(0, _do)

    def _set_status(self, t):
        self.after(0, lambda: self.status_var.set(t))

    def _set_info(self, label=None, conf=None):
        t = f"Prediction: {label or '—'}"
        if conf is not None: t += f"  ({conf:.1%})"
        self.after(0, lambda: self.info_text.set(t))

    def _update_preview(self, pil):
        def _do():
            w = 360
            h = pil.height * w // pil.width
            self.current_photo = ImageTk.PhotoImage(pil.resize((w, h)))
            self.preview_label.config(image=self.current_photo, text="")

        self.after(0, _do)

    def _set_progress(self, val, mx=None):
        def _do():
            if mx is not None: self.progress["maximum"] = mx
            self.progress["value"] = val
            top = float(self.progress["maximum"]) or 1.0
            self.percent_var.set(f"{int(round(float(val) / top * 100))}%")

        self.after(0, _do)

    def _set_eta(self, s):
        self.after(0, lambda: self.eta_var.set("ETA: " + (format_duration(s) if s is not None else "--:--")))

    def _set_busy(self, busy=True):
        def _do():
            st = "disabled" if busy else "normal"
            self.classify_folder_btn.config(state=st)
            self.path_entry.config(state=st)

        self.after(0, _do)

    def _refresh_ui(self):
        def _do():
            if self.model and self.model.classes:
                self.custom_list_var.set("Classes: " + ", ".join(self.model.classes))
            else:
                self.custom_list_var.set("(none)")
            if self.model and self.model.trained:
                self.train_status_var.set(f"Trained ({len(self.model.classes)} classes)")
            else:
                self.train_status_var.set("Not trained")

        self.after(0, _do)

    def _browse_file(self):
        f = filedialog.askopenfilename(title="Select an image",
                        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All", "*.*")])
        if f: self.path_var.set(f)

    def _choose_folder(self):
        d = filedialog.askdirectory(title="Select images folder")
        if d:
            self.selected_folder = d
            self.folder_label_var.set(d)
            self._log(f"Selected folder: {d}")

    def _start_single(self):
        self._log("Classify — not implemented yet")

    def _start_folder(self):
        self._log("Classify folder — not implemented yet")

    def _add_class(self):
        name = simpledialog.askstring("New Class", "Class name:", parent=self)
        if not name: return
        name = name.strip()

        class_dir = os.path.join(CUSTOM_DATA_DIR, name)
        os.makedirs(class_dir, exist_ok=True)

        folder = filedialog.askdirectory(title=f"Select training images for '{name}'")
        count = 0
        if folder:
            for f in os.listdir(folder):
                if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                    shutil.copy2(os.path.join(folder, f), os.path.join(class_dir, f))
                    count += 1

        self.model.add_class(name)
        self.model.save()
        self._refresh_ui()
        self._log(f"Added '{name}' with {count} images.")

    def _remove_class(self):
        names = list(self.model.classes.keys())
        if not names: return

        win = tk.Toplevel(self)
        win.title("Remove Class")
        win.geometry("300x200")
        win.configure(bg=self.surface)
        ttk.Label(win, text="Select class to remove:").pack(pady=(10, 4))
        lb = tk.Listbox(win, bg=self.card, fg=self.text, font=self.font_normal, selectmode=tk.SINGLE)
        for n in names: lb.insert(tk.END, n)
        lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        def do_remove():
            sel = lb.curselection()
            if sel:
                chosen = names[sel[0]]
            self.model.remove_class(chosen)
            self.model.save()
            self._refresh_ui()
            self._log(f"Removed '{chosen}'.")
            win.destroy()

        ttk.Button(win, text="Remove", style="Danger.TButton", command=do_remove).pack(pady=(4, 10))

    def _start_training(self):
        self._log("Train — not implemented yet")

    def _clear_log(self):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.insert(tk.END, "Ready.\n")
        self.log_box.config(state=tk.DISABLED)
        self.preview_label.config(image="", text="No image")
        self.info_text.set("Prediction: —")
        self._set_progress(0, 1)
        self._set_eta(None)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ModernClassifierApp().mainloop()
