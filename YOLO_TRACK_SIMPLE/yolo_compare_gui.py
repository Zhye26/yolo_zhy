#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
from pathlib import Path
from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    DoubleVar,
    Entry,
    Frame,
    IntVar,
    Label,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


class YoloCompareGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("YOLO Weights Video Compare")
        self.root.geometry("1380x840")

        base = Path(__file__).resolve().parent
        self.video_path = StringVar(value="")
        self.left_model_path = StringVar(value=str(base / "weights" / "wjh.pt"))
        self.right_model_path = StringVar(value=str(base / "weights" / "zhuyu.pt"))
        self.conf = DoubleVar(value=0.25)
        self.iou = DoubleVar(value=0.45)
        self.imgsz = IntVar(value=640)
        self.frame_skip = IntVar(value=1)
        self.save_video = BooleanVar(value=False)

        self.status_text = StringVar(value="Idle")
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.current_photo = None

        self._build_ui()

    def _build_ui(self) -> None:
        left = Frame(self.root, padx=10, pady=10)
        left.pack(side="left", fill="y")
        right = Frame(self.root, padx=10, pady=10)
        right.pack(side="left", fill="both", expand=True)

        self._entry_row(left, "Video", self.video_path, btn_text="Browse", btn_cmd=self.pick_video, width=38)
        self._entry_row(left, "Left Model", self.left_model_path, btn_text="Browse", btn_cmd=self.pick_left_model, width=38)
        self._entry_row(left, "Right Model", self.right_model_path, btn_text="Browse", btn_cmd=self.pick_right_model, width=38)
        self._entry_row(left, "conf", self.conf)
        self._entry_row(left, "iou", self.iou)
        self._entry_row(left, "imgsz", self.imgsz)
        self._entry_row(left, "frame_skip", self.frame_skip)

        ctrl = Frame(left)
        ctrl.pack(fill="x", pady=(8, 6))
        Button(ctrl, text="Start", width=10, command=self.start).pack(side="left", padx=(0, 4))
        Button(ctrl, text="Pause/Resume", width=12, command=self.toggle_pause).pack(side="left", padx=(0, 4))
        Button(ctrl, text="Stop", width=10, command=self.stop).pack(side="left")

        Checkbutton(left, text="Save compare video", variable=self.save_video).pack(anchor="w", pady=4)

        Label(left, textvariable=self.status_text, wraplength=360, justify="left").pack(fill="x", pady=(8, 0))

        self.video_label = Label(right, bg="black")
        self.video_label.pack(fill="both", expand=True)

    def _entry_row(self, parent: Frame, label: str, var, btn_text: str | None = None, btn_cmd=None, width: int = 24) -> None:
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        Label(row, text=label, width=14, anchor="w").pack(side="left")
        Entry(row, textvariable=var, width=width).pack(side="left")
        if btn_text and btn_cmd:
            Button(row, text=btn_text, width=8, command=btn_cmd).pack(side="left", padx=(4, 0))

    def pick_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
        )
        if p:
            self.video_path.set(p)

    def pick_left_model(self) -> None:
        p = filedialog.askopenfilename(title="Select left model", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if p:
            self.left_model_path.set(p)

    def pick_right_model(self) -> None:
        p = filedialog.askopenfilename(title="Select right model", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if p:
            self.right_model_path.set(p)

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Info", "Worker is running.")
            return

        video = Path(self.video_path.get().strip())
        left_model = Path(self.left_model_path.get().strip())
        right_model = Path(self.right_model_path.get().strip())
        if not video.exists():
            messagebox.showerror("Error", f"Video not found: {video}")
            return
        if not left_model.exists() or not right_model.exists():
            messagebox.showerror("Error", "Model path not found.")
            return

        self.stop_flag.clear()
        self.pause_flag.clear()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def toggle_pause(self) -> None:
        if self.pause_flag.is_set():
            self.pause_flag.clear()
            self.status_text.set("Resumed")
        else:
            self.pause_flag.set()
            self.status_text.set("Paused")

    def stop(self) -> None:
        self.stop_flag.set()
        self.pause_flag.clear()
        self.status_text.set("Stopping...")

    def _run(self) -> None:
        video_path = Path(self.video_path.get().strip())
        left_model_path = Path(self.left_model_path.get().strip())
        right_model_path = Path(self.right_model_path.get().strip())
        conf = float(self.conf.get())
        iou = float(self.iou.get())
        imgsz = int(self.imgsz.get())
        frame_skip = max(1, int(self.frame_skip.get()))

        try:
            left_model = YOLO(str(left_model_path))
            right_model = YOLO(str(right_model_path))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Load model failed:\n{e}"))
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Error", f"Open video failed:\n{video_path}"))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if bool(self.save_video.get()):
            out_dir = Path("runs") / "compare"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{video_path.stem}_compare_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width * 2, height))
            self.root.after(0, lambda: self.status_text.set(f"Saving to: {out_path}"))

        frame_id = 0
        shown = 0
        while True:
            if self.stop_flag.is_set():
                break
            if self.pause_flag.is_set():
                time.sleep(0.03)
                continue

            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            left_res = left_model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            right_res = right_model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

            left_img = left_res.plot()
            right_img = right_res.plot()
            cv2.putText(left_img, f"{left_model_path.name}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(right_img, f"{right_model_path.name}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            merged = cv2.hconcat([left_img, right_img])

            shown += 1
            if writer is not None:
                writer.write(merged)

            rgb = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            max_w, max_h = 980, 760
            scale = min(max_w / image.width, max_h / image.height, 1.0)
            if scale < 1:
                image = image.resize((int(image.width * scale), int(image.height * scale)))
            photo = ImageTk.PhotoImage(image=image)
            pct = (frame_id / total * 100.0) if total > 0 else 0.0
            status = f"Frame {frame_id}/{total} ({pct:.1f}%), shown={shown}"
            self.root.after(0, self._update_frame, photo, status)

        cap.release()
        if writer is not None:
            writer.release()
        self.root.after(0, lambda: self.status_text.set("Stopped" if self.stop_flag.is_set() else "Finished"))

    def _update_frame(self, photo: ImageTk.PhotoImage, status: str) -> None:
        self.current_photo = photo
        self.video_label.configure(image=photo)
        self.status_text.set(status)


def main() -> None:
    root = Tk()
    app = YoloCompareGui(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
