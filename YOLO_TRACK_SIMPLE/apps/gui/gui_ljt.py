#!/usr/bin/env python3
"""
Standalone Tkinter GUI for real-time yolov8n-only overload validation.

This GUI intentionally does not use the original Flask web app, database,
pipeline, or trained model code. It reuses only this method package's
root pipeline_ljt.py.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from pathlib import Path
from tkinter import (
    BooleanVar,
    Button,
    Canvas,
    Checkbutton,
    DoubleVar,
    Entry,
    Frame,
    IntVar,
    Label,
    Listbox,
    OptionMenu,
    Scrollbar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from typing import Dict, List

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from methods.mot_trackers_ljt import AVAILABLE_TRACKERS

from core.pipeline.pipeline_ljt import (
    ASSOCIATION_MOT_TRACKERS,
    MOTORCYCLE_CLASS_ID,
    PERSON_CLASS_ID,
    FrameVehicleResult,
    create_vehicle_tracker,
    draw_frame,
    match_people_to_vehicles,
    result_to_detections,
    sync_cuda_if_available,
    uses_association_scene,
)


class Yolov8nOverloadGuiLjt:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("YOLOv8n Overload MOT Compare - ljt")
        self.root.geometry("1180x820")

        self.video_paths: List[Path] = []
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.ui_queue: queue.Queue = queue.Queue(maxsize=3)
        self.model = None
        self.current_photo = None
        self.track_result_rows: List[str] = []

        project_root = Path(__file__).resolve().parents[2]
        default_model = project_root / "weights" / "yolov8n.pt"
        self.model_path = StringVar(value=str(default_model))
        self.detect_classes = StringVar(value="0,3")
        self.conf = DoubleVar(value=0.25)
        self.iou = DoubleVar(value=0.45)
        self.imgsz = IntVar(value=640)
        self.match_thresh = DoubleVar(value=1.05)
        self.confirm_frames = IntVar(value=2)
        self.tracker = StringVar(value="association")
        self.track_iou = DoubleVar(value=0.18)
        self.max_missed = IntVar(value=6)
        self.byte_track_thresh = DoubleVar(value=0.25)
        self.byte_match_thresh = DoubleVar(value=0.8)
        self.byte_track_buffer = IntVar(value=30)
        self.mot_min_hits = IntVar(value=3)
        self.reid_weights = StringVar(
            value=str(project_root / "weights" / "osnet_x0_25_msmt17.pt")
        )
        self.reid_device = StringVar(value="cpu")
        self.reid_fp16 = BooleanVar(value=False)
        self.association_min_hits = IntVar(value=4)
        self.association_lock_frames = IntVar(value=20)
        self.association_unbind_frames = IntVar(value=15)
        self.association_switch_margin = DoubleVar(value=0.35)
        self.min_area = DoubleVar(value=20.0)
        self.frame_stride = IntVar(value=1)
        self.save_video = BooleanVar(value=False)

        self.status_text = StringVar(value="Idle")
        self.stats_text = StringVar(value="No video loaded")

        self._build_layout()
        self.root.after(30, self._poll_ui_queue)

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(0, weight=0, minsize=430)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        left_outer = Frame(self.root)
        left_outer.grid(row=0, column=0, sticky="nsw")
        left_outer.grid_rowconfigure(0, weight=1)
        left_outer.grid_columnconfigure(0, weight=1)

        left_canvas = Canvas(left_outer, width=420, highlightthickness=0)
        left_scroll = Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scroll.grid(row=0, column=1, sticky="ns")

        left = Frame(left_canvas, padx=10, pady=10)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _sync_canvas_width(event) -> None:
            left_canvas.itemconfigure(left_window, width=event.width)

        left.bind("<Configure>", _sync_scroll_region)
        left_canvas.bind("<Configure>", _sync_canvas_width)
        left_canvas.bind_all("<MouseWheel>", lambda event: left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        right = Frame(self.root, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        file_actions = Frame(left)
        file_actions.pack(fill="x", pady=(0, 4))
        Button(file_actions, text="Add Videos", command=self.add_videos, width=18).pack(side="left", padx=(0, 6))
        Button(file_actions, text="Clear List", command=self.clear_videos, width=18).pack(side="left")

        run_actions = Frame(left)
        run_actions.pack(fill="x", pady=(2, 6))
        Button(run_actions, text="Start", command=self.start, width=12).pack(side="left", padx=(0, 5))
        Button(run_actions, text="Pause / Resume", command=self.toggle_pause, width=16).pack(side="left", padx=(0, 5))
        Button(run_actions, text="Stop", command=self.stop, width=10).pack(side="left")

        Label(left, textvariable=self.status_text, wraplength=360, justify="left").pack(
            fill="x", pady=(0, 8), anchor="w"
        )

        self.video_list = Listbox(left, width=46, height=8)
        self.video_list.pack(pady=8)

        self._labeled_entry(left, "Model", self.model_path)
        self._labeled_entry(left, "detect_classes", self.detect_classes)
        self._labeled_entry(left, "conf", self.conf)
        self._labeled_entry(left, "iou", self.iou)
        self._labeled_entry(left, "imgsz", self.imgsz)
        self._labeled_entry(left, "match_thresh", self.match_thresh)
        self._labeled_entry(left, "confirm_frames", self.confirm_frames)
        self._labeled_option(
            left,
            "tracker",
            self.tracker,
            ["association", "iou", "auto", *ASSOCIATION_MOT_TRACKERS, *sorted(AVAILABLE_TRACKERS)],
        )
        self._labeled_entry(left, "track_iou", self.track_iou)
        self._labeled_entry(left, "max_missed", self.max_missed)
        self._labeled_entry(left, "byte_track_thresh", self.byte_track_thresh)
        self._labeled_entry(left, "byte_match_thresh", self.byte_match_thresh)
        self._labeled_entry(left, "byte_track_buffer", self.byte_track_buffer)
        self._labeled_entry(left, "mot_min_hits", self.mot_min_hits)
        self._labeled_entry(left, "reid_weights", self.reid_weights)
        self._labeled_entry(left, "reid_device", self.reid_device)
        self._labeled_entry(left, "assoc_min_hits", self.association_min_hits)
        self._labeled_entry(left, "assoc_lock", self.association_lock_frames)
        self._labeled_entry(left, "assoc_unbind", self.association_unbind_frames)
        self._labeled_entry(left, "assoc_switch", self.association_switch_margin)
        self._labeled_entry(left, "min_area", self.min_area)
        self._labeled_entry(left, "frame_stride", self.frame_stride)

        Checkbutton(left, text="ReID fp16", variable=self.reid_fp16).pack(anchor="w", pady=(4, 0))
        Checkbutton(left, text="Save annotated videos", variable=self.save_video).pack(anchor="w", pady=6)

        self.video_label = Label(right, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        Label(right, textvariable=self.stats_text, justify="left", anchor="w").grid(
            row=1, column=0, sticky="ew", pady=(8, 4)
        )

        Label(right, text="Track Results", anchor="w").grid(row=2, column=0, sticky="ew")
        self.track_result_list = Listbox(right, height=8, font=("Menlo", 12))
        self.track_result_list.grid(row=3, column=0, sticky="ew")

    def _labeled_entry(self, parent: Frame, label: str, variable) -> None:
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        Label(row, text=label, width=14, anchor="w").pack(side="left")
        Entry(row, textvariable=variable, width=24).pack(side="right")

    def _labeled_option(self, parent: Frame, label: str, variable, options: List[str]) -> None:
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        Label(row, text=label, width=14, anchor="w").pack(side="left")
        menu = OptionMenu(row, variable, *options)
        menu.configure(width=20)
        menu.pack(side="right")

    def add_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        for item in paths:
            path = Path(item)
            if path not in self.video_paths:
                self.video_paths.append(path)
                self.video_list.insert("end", str(path))

    def clear_videos(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Stop processing before clearing the list.")
            return
        self.video_paths.clear()
        self.video_list.delete(0, "end")
        self.stats_text.set("No video loaded")
        self.track_result_list.delete(0, "end")

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Processing is already running.")
            return
        if not self.video_paths:
            messagebox.showwarning("No videos", "Add one or more videos first.")
            return

        self.stop_event.clear()
        self.pause_event.clear()
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()
        self.status_text.set("Starting...")

    def toggle_pause(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.status_text.set("Running")
        else:
            self.pause_event.set()
            self.status_text.set("Paused")

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.status_text.set("Stopping...")

    def _load_model(self) -> YOLO:
        if self.model is None:
            self.model = YOLO(self.model_path.get())
        return self.model

    def _parse_detect_classes(self) -> List[int]:
        raw = self.detect_classes.get().strip()
        if not raw:
            return [PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID]
        class_ids: List[int] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            class_ids.append(int(item))
        return class_ids or [PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID]

    def _run_worker(self) -> None:
        try:
            model = self._load_model()
            for index, video_path in enumerate(self.video_paths, start=1):
                if self.stop_event.is_set():
                    break
                self._process_video(model, video_path, index, len(self.video_paths))
            self._put_ui({"type": "status", "text": "Done" if not self.stop_event.is_set() else "Stopped"})
        except Exception as exc:
            self._put_ui({"type": "error", "text": str(exc)})

    def _process_video(self, model: YOLO, video_path: Path, video_index: int, video_total: int) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._put_ui({"type": "status", "text": f"Failed to open: {video_path}"})
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detect_classes = self._parse_detect_classes()

        output_dir = video_path.parent
        tracker_slug = self.tracker.get().strip().lower().replace(" ", "_")
        stem = f"{video_path.stem}-yolov8n-overload-{tracker_slug}-ljt"
        frame_csv_path = output_dir / f"{stem}_frames.csv"
        summary_csv_path = output_dir / f"{stem}_summary.csv"
        video_out_path = output_dir / f"{stem}.mp4"
        writer = None
        if self.save_video.get():
            writer = cv2.VideoWriter(
                str(video_out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps / max(1, self.frame_stride.get()),
                (frame_w, frame_h),
            )

        ok, first_frame = cap.read()
        if ok:
            model.predict(
                first_frame,
                classes=detect_classes,
                conf=self.conf.get(),
                iou=self.iou.get(),
                imgsz=self.imgsz.get(),
                verbose=False,
            )
            sync_cuda_if_available()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        tracker_args = type(
            "TrackerArgs",
            (),
            {
                "tracker": self.tracker.get(),
                "byte_track_thresh": self.byte_track_thresh.get(),
                "byte_match_thresh": self.byte_match_thresh.get(),
                "byte_track_buffer": self.byte_track_buffer.get(),
                "mot_min_hits": self.mot_min_hits.get(),
                "reid_weights": self.reid_weights.get(),
                "reid_device": self.reid_device.get(),
                "reid_fp16": self.reid_fp16.get(),
                "confirm_frames": self.confirm_frames.get(),
                "max_missed": self.max_missed.get(),
                "track_iou": self.track_iou.get(),
                "association_min_hits": self.association_min_hits.get(),
                "association_lock_frames": self.association_lock_frames.get(),
                "association_unbind_frames": self.association_unbind_frames.get(),
                "association_switch_margin": self.association_switch_margin.get(),
            },
        )()
        tracker, tracker_name = create_vehicle_tracker(tracker_args)

        rows: List[Dict[str, object]] = []
        track_summary: Dict[str, Dict[str, int | bool]] = {}
        timing_ms: List[float] = []
        processed = 0
        frame_index = -1
        frames_with_raw = 0
        frames_with_confirmed = 0
        total_people = 0
        total_vehicles = 0
        start_wall = time.perf_counter()

        while not self.stop_event.is_set():
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)

            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % max(1, self.frame_stride.get()) != 0:
                continue

            timestamp_sec = frame_index / source_fps if source_fps > 0 else 0.0
            sync_cuda_if_available()
            start = time.perf_counter()
            result = model.predict(
                frame,
                classes=detect_classes,
                conf=self.conf.get(),
                iou=self.iou.get(),
                imgsz=self.imgsz.get(),
                verbose=False,
            )[0]
            sync_cuda_if_available()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timing_ms.append(elapsed_ms)

            detections = result_to_detections(result, min_area=self.min_area.get())
            people = [det for det in detections if det.class_id == PERSON_CLASS_ID]
            vehicles = [det for det in detections if det.class_id == MOTORCYCLE_CLASS_ID]
            total_people += len(people)
            total_vehicles += len(vehicles)

            if uses_association_scene(tracker_name):
                track_matches, grouped_people, match_scores = tracker.update_scene(
                    people,
                    vehicles,
                    frame,
                    self.match_thresh.get(),
                )
            else:
                grouped_people, match_scores = match_people_to_vehicles(people, vehicles, self.match_thresh.get())
                rider_counts = {idx: len(grouped_people[idx]) for idx in range(len(vehicles))}
                if tracker_name in AVAILABLE_TRACKERS:
                    track_matches = tracker.update(vehicles, rider_counts, frame, grouped_people, match_scores)
                else:
                    track_matches = tracker.update(vehicles, rider_counts, grouped_people, match_scores)

            frame_results: List[FrameVehicleResult] = []
            has_raw = False
            has_confirmed = False
            for vehicle_idx, vehicle in enumerate(vehicles):
                track_id = track_matches[vehicle_idx]
                matched_people = grouped_people[vehicle_idx]
                raw_overload = len(matched_people) >= 2
                if hasattr(tracker, "get_overload_status"):
                    overload_status = str(tracker.get_overload_status(track_id))
                    confirmed = overload_status == "CONFIRMED"
                else:
                    confirmed = tracker.is_confirmed(track_id)
                    overload_status = "CONFIRMED" if confirmed else "SUSPECTED" if raw_overload else "NORMAL"
                if str(track_id).startswith("U"):
                    raw_overload = False
                    confirmed = False
                    overload_status = "UNCERTAIN"
                has_raw = has_raw or raw_overload
                has_confirmed = has_confirmed or confirmed
                self._update_track_summary(
                    track_summary,
                    track_id=track_id,
                    raw_overload=raw_overload,
                    confirmed_overload=confirmed,
                    rider_count=len(matched_people),
                    frame_index=frame_index,
                    overload_status=overload_status,
                )
                frame_results.append(
                    FrameVehicleResult(
                        track_id=track_id,
                        detection=vehicle,
                        matched_people=matched_people,
                        match_scores=match_scores[vehicle_idx],
                        raw_overload=raw_overload,
                        confirmed_overload=confirmed,
                    )
                )
                rows.append(
                    {
                        "video": str(video_path),
                        "frame": frame_index,
                        "timestamp_sec": f"{timestamp_sec:.3f}",
                        "vehicle_track_id": track_id,
                        "vehicle_class": vehicle.class_name,
                        "vehicle_conf": f"{vehicle.confidence:.3f}",
                        "vehicle_bbox": " ".join(f"{v:.1f}" for v in vehicle.bbox),
                        "matched_person_ids": " ".join(getattr(person, "stable_id", "") for person in matched_people),
                        "matched_person_count": len(matched_people),
                        "match_scores": " ".join(f"{score:.3f}" for score in match_scores[vehicle_idx]),
                        "raw_overload": int(raw_overload),
                        "confirmed_overload": int(confirmed),
                        "elapsed_ms": f"{elapsed_ms:.3f}",
                        "fps": f"{1000.0 / elapsed_ms:.3f}" if elapsed_ms > 0 else "",
                    }
                )

            if has_raw:
                frames_with_raw += 1
            if has_confirmed:
                frames_with_confirmed += 1

            avg_ms_30 = sum(timing_ms[-30:]) / min(len(timing_ms), 30)
            inst_fps = 1000.0 / avg_ms_30 if avg_ms_30 > 0 else 0.0
            annotated = draw_frame(frame, people, frame_results, frame_index, timestamp_sec, f"fps={inst_fps:.1f}")
            if writer is not None:
                writer.write(annotated)

            processed += 1
            wall_elapsed = max(1e-6, time.perf_counter() - start_wall)
            wall_fps = processed / wall_elapsed
            self._put_ui(
                {
                    "type": "frame",
                    "frame": annotated,
                    "status": f"Video {video_index}/{video_total}: {video_path.name}",
                    "stats": (
                        f"processed={processed}/{total_frames}  wall_fps={wall_fps:.1f}  "
                        f"infer_fps={inst_fps:.1f}  tracker={tracker_name}  people={total_people}  two_wheelers={total_vehicles}  "
                        f"raw_frames={frames_with_raw}  confirmed_frames={frames_with_confirmed}  "
                        f"confirmed_tracks={len(tracker.confirmed_track_ids)}"
                    ),
                    "track_results": self._format_track_summary(track_summary),
                }
            )

        cap.release()
        if writer is not None:
            writer.release()

        self._write_frame_csv(frame_csv_path, rows)
        avg_ms = sum(timing_ms) / len(timing_ms) if timing_ms else 0.0
        self._write_summary_csv(
            summary_csv_path,
            video_path=video_path,
            processed=processed,
            total_frames=total_frames,
            source_fps=source_fps,
            total_people=total_people,
            total_vehicles=total_vehicles,
            frames_with_raw=frames_with_raw,
            frames_with_confirmed=frames_with_confirmed,
            confirmed_tracks=len(tracker.confirmed_track_ids),
            avg_ms=avg_ms,
            saved_video=str(video_out_path) if writer is not None else "",
            tracker_name=tracker_name,
            detect_classes=detect_classes,
        )
        self._put_ui({"type": "status", "text": f"Saved CSV: {summary_csv_path}"})

    def _update_track_summary(
        self,
        track_summary: Dict[str, Dict[str, int | bool]],
        track_id,
        raw_overload: bool,
        confirmed_overload: bool,
        rider_count: int,
        frame_index: int,
        overload_status: str,
    ) -> None:
        key = str(track_id)
        state = track_summary.setdefault(
            key,
            {
                "seen_frames": 0,
                "raw_frames": 0,
                "confirmed_frames": 0,
                "high_conf_overload_ratio": 0.0,
                "moving_overload_ratio": 0.0,
                "avg_speed_px": 0.0,
                "ever_confirmed": False,
                "last_status": "NORMAL",
                "last_rider_count": 0,
                "last_frame": 0,
            },
        )
        state["seen_frames"] = int(state["seen_frames"]) + 1
        state["raw_frames"] = int(state["raw_frames"]) + int(raw_overload)
        state["confirmed_frames"] = int(state["confirmed_frames"]) + int(confirmed_overload)
        state["ever_confirmed"] = bool(state["ever_confirmed"]) or confirmed_overload
        state["last_status"] = overload_status
        state["last_rider_count"] = rider_count
        state["last_frame"] = frame_index
        state["raw_overload_ratio"] = int(state["raw_frames"]) / max(1, int(state["seen_frames"]))

    def _format_track_summary(self, track_summary: Dict[str, Dict[str, int | bool]]) -> List[str]:
        lines = ["ID        result      seen  raw%  conf  riders  last"]
        def sort_key(item):
            track_id, state = item
            return (not bool(state["ever_confirmed"]), -int(state["confirmed_frames"]), str(track_id))

        for track_id, state in sorted(track_summary.items(), key=sort_key):
            if str(track_id).startswith("U"):
                result = "UNCERTAIN"
            elif bool(state["ever_confirmed"]):
                result = "CONFIRMED"
            elif str(state.get("last_status", "")) == "SUSPECTED" or int(state["raw_frames"]) > 0:
                result = "SUSPECTED"
            else:
                result = "NORMAL"
            lines.append(
                f"{track_id:<9} {result:<10} "
                f"{int(state['seen_frames']):>4} "
                f"{float(state.get('raw_overload_ratio', 0.0)) * 100:>4.0f} "
                f"{int(state['confirmed_frames']):>5} "
                f"{int(state['last_rider_count']):>6} "
                f"{int(state['last_frame']):>5}"
            )
        return lines[:101]

    def _write_frame_csv(self, path: Path, rows: List[Dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "video",
                "frame",
                "timestamp_sec",
                "vehicle_track_id",
                "vehicle_class",
                "vehicle_conf",
                "vehicle_bbox",
                "matched_person_ids",
                "matched_person_count",
                "match_scores",
                "raw_overload",
                "confirmed_overload",
                "elapsed_ms",
                "fps",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary_csv(
        self,
        path: Path,
        video_path: Path,
        processed: int,
        total_frames: int,
        source_fps: float,
        total_people: int,
        total_vehicles: int,
        frames_with_raw: int,
        frames_with_confirmed: int,
        confirmed_tracks: int,
        avg_ms: float,
        saved_video: str,
        tracker_name: str,
        detect_classes: List[int],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "video",
                "model",
                "detect_classes",
                "tracker",
                "processed_frames",
                "source_frames",
                "source_fps",
                "imgsz",
                "conf",
                "match_thresh",
                "confirm_frames",
                "mot_min_hits",
                "reid_weights",
                "reid_device",
                "reid_fp16",
                "total_people_detections",
                "total_two_wheeler_detections",
                "frames_with_raw_overload",
                "frames_with_confirmed_overload",
                "confirmed_overload_tracks",
                "avg_elapsed_ms",
                "avg_fps",
                "saved_video",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "video": str(video_path),
                    "model": self.model_path.get(),
                    "detect_classes": ",".join(str(item) for item in detect_classes),
                    "tracker": tracker_name,
                    "processed_frames": processed,
                    "source_frames": total_frames,
                    "source_fps": f"{source_fps:.3f}",
                    "imgsz": self.imgsz.get(),
                    "conf": self.conf.get(),
                    "match_thresh": self.match_thresh.get(),
                    "confirm_frames": self.confirm_frames.get(),
                    "mot_min_hits": self.mot_min_hits.get(),
                    "reid_weights": self.reid_weights.get(),
                    "reid_device": self.reid_device.get(),
                    "reid_fp16": int(self.reid_fp16.get()),
                                            "total_people_detections": total_people,
                    "total_two_wheeler_detections": total_vehicles,
                    "frames_with_raw_overload": frames_with_raw,
                    "frames_with_confirmed_overload": frames_with_confirmed,
                    "confirmed_overload_tracks": confirmed_tracks,
                    "avg_elapsed_ms": f"{avg_ms:.3f}",
                    "avg_fps": f"{1000.0 / avg_ms:.3f}" if avg_ms > 0 else "",
                    "saved_video": saved_video,
                }
            )

    def _put_ui(self, item: Dict[str, object]) -> None:
        try:
            self.ui_queue.put_nowait(item)
        except queue.Full:
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                pass
            self.ui_queue.put_nowait(item)

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                item = self.ui_queue.get_nowait()
                if item["type"] == "frame":
                    self.status_text.set(str(item["status"]))
                    self.stats_text.set(str(item["stats"]))
                    self._show_track_results(item.get("track_results", []))
                    self._show_frame(item["frame"])
                elif item["type"] == "status":
                    self.status_text.set(str(item["text"]))
                elif item["type"] == "error":
                    self.status_text.set("Error")
                    messagebox.showerror("Processing error", str(item["text"]))
        except queue.Empty:
            pass
        self.root.after(30, self._poll_ui_queue)

    def _show_track_results(self, rows) -> None:
        self.track_result_list.delete(0, "end")
        for row in rows:
            self.track_result_list.insert("end", str(row))

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        label_w = max(640, self.video_label.winfo_width())
        label_h = max(360, self.video_label.winfo_height())
        img.thumbnail((label_w, label_h))
        self.current_photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.current_photo)


def main() -> None:
    root = Tk()
    Yolov8nOverloadGuiLjt(root)
    root.mainloop()


if __name__ == "__main__":
    main()
