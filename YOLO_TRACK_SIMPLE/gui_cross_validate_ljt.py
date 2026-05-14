#!/usr/bin/env python3
from __future__ import annotations

import csv
import queue
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
    Text,
    Tk,
    filedialog,
    messagebox,
)
from typing import Dict, List

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from gui_wjh_ljt import (
    HEAD_CLASS_ID,
    HELMET_CLASS_ID,
    RIDER_CLASS_ID,
    TARGET_CLASSES as WJH_CLASSES,
    build_person_evidence_for_rider,
)
from pipeline_ljt import (
    MOTORCYCLE_CLASS_ID,
    PERSON_CLASS_ID,
    Detection,
    iou as bbox_iou,
    match_people_to_vehicles,
    result_to_detections,
    sync_cuda_if_available,
)


YOLOV8N_CLASSES = [PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID]


class IouIdTracker:
    def __init__(self, iou_thresh: float = 0.25, max_missed: int = 10, prefix: str = "T"):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.prefix = prefix
        self.next_id = 1
        self.tracks: Dict[str, Dict[str, object]] = {}

    def update(self, detections: List[Detection]) -> Dict[int, str]:
        matches: Dict[int, str] = {}
        used_tracks: set[str] = set()
        candidates: List[tuple[float, str, int]] = []
        for track_id, state in self.tracks.items():
            if int(state["missed"]) > self.max_missed:
                continue
            for det_idx, det in enumerate(detections):
                score = bbox_iou(state["bbox"], det.bbox)
                if score >= self.iou_thresh:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(reverse=True)

        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in matches:
                continue
            used_tracks.add(track_id)
            matches[det_idx] = track_id

        for track_id, state in list(self.tracks.items()):
            if track_id not in used_tracks:
                state["missed"] = int(state["missed"]) + 1

        for det_idx, det in enumerate(detections):
            track_id = matches.get(det_idx)
            if track_id is None:
                track_id = f"{self.prefix}{self.next_id:03d}"
                self.next_id += 1
                matches[det_idx] = track_id
            self.tracks[track_id] = {"bbox": det.bbox, "missed": 0, "confidence": det.confidence}
        return matches


class CrossValidateGuiLjt:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Cross Validate: wjh.pt + yolov8n.pt")
        self.root.geometry("1280x820")

        base = Path(__file__).resolve().parent
        self.video_path = StringVar(value="")
        self.wjh_model_path = StringVar(value=str(base / "weights" / "wjh.pt"))
        self.yolo_model_path = StringVar(value=str(base / "weights" / "yolov8n.pt"))
        self.conf = DoubleVar(value=0.25)
        self.head_conf = DoubleVar(value=0.25)
        self.helmet_conf = DoubleVar(value=0.45)
        self.iou = DoubleVar(value=0.45)
        self.imgsz = IntVar(value=640)
        self.wjh_min_area = DoubleVar(value=12.0)
        self.yolo_min_area = DoubleVar(value=20.0)
        self.yolo_match_thresh = DoubleVar(value=1.05)
        self.cross_iou_thresh = DoubleVar(value=0.20)
        self.frame_stride = IntVar(value=1)
        self.save_video = BooleanVar(value=False)

        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.ui_queue: queue.Queue = queue.Queue(maxsize=3)
        self.wjh_photo = None
        self.yolo_photo = None
        self.last_wjh_frame = None
        self.last_yolo_frame = None

        self.roi_rect: tuple[int, int, int, int] | None = None
        self.roi_drag_start: tuple[int, int] | None = None
        self.roi_drag_current: tuple[int, int] | None = None
        self.active_roi_view = "wjh"
        self.display_meta: Dict[str, Dict[str, tuple[int, int]]] = {
            "wjh": {"image_size": (0, 0), "offset": (0, 0), "frame_shape": (0, 0)},
            "yolo": {"image_size": (0, 0), "offset": (0, 0), "frame_shape": (0, 0)},
        }

        self.status_text = StringVar(value="Add a video, draw ROI if needed, then Start")
        self.stats_text = StringVar(value="No video loaded")
        self.result_rows: List[str] = []
        self._build_layout()
        self.root.after(30, self._poll_ui_queue)

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(0, weight=0, minsize=410)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        left = Frame(self.root, padx=10, pady=10)
        left.grid(row=0, column=0, sticky="nsw")
        right = Frame(self.root, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)
        right.grid_rowconfigure(1, weight=1)

        Button(left, text="Add Video", command=self.pick_video, width=16).pack(anchor="w", pady=(0, 4))
        Label(left, textvariable=self.status_text, wraplength=360, justify="left").pack(fill="x", pady=(0, 8), anchor="w")

        run_actions = Frame(left)
        run_actions.pack(fill="x", pady=(0, 6))
        Button(run_actions, text="Start", command=self.start, width=10).pack(side="left", padx=(0, 4))
        Button(run_actions, text="Pause", command=self.toggle_pause, width=10).pack(side="left", padx=(0, 4))
        Button(run_actions, text="Stop", command=self.stop, width=10).pack(side="left")

        roi_actions = Frame(left)
        roi_actions.pack(fill="x", pady=(0, 8))
        Button(roi_actions, text="Clear ROI", command=self.clear_roi, width=10).pack(side="left", padx=(0, 4))
        Label(roi_actions, text="Drag on preview before Start").pack(side="left")

        for label, var in (
            ("Video", self.video_path),
            ("wjh_model", self.wjh_model_path),
            ("yolo_model", self.yolo_model_path),
            ("conf", self.conf),
            ("head_conf", self.head_conf),
            ("helmet_conf", self.helmet_conf),
            ("iou", self.iou),
            ("imgsz", self.imgsz),
            ("wjh_min_area", self.wjh_min_area),
            ("yolo_min_area", self.yolo_min_area),
            ("yolo_match", self.yolo_match_thresh),
            ("cross_iou", self.cross_iou_thresh),
            ("frame_stride", self.frame_stride),
        ):
            self._entry_row(left, label, var)
        Checkbutton(left, text="Save annotated video", variable=self.save_video).pack(anchor="w", pady=6)

        Label(right, text="wjh.pt rider/head/helmet", anchor="w").grid(row=0, column=0, sticky="ew")
        Label(right, text="yolov8n.pt person/motorcycle", anchor="w").grid(row=0, column=1, sticky="ew")
        self.wjh_label = Label(right, bg="black")
        self.yolo_label = Label(right, bg="black")
        self.wjh_label.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.yolo_label.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        for view, label in (("wjh", self.wjh_label), ("yolo", self.yolo_label)):
            label.bind("<ButtonPress-1>", lambda event, v=view: self._on_roi_press(event, v))
            label.bind("<B1-Motion>", lambda event, v=view: self._on_roi_drag(event, v))
            label.bind("<ButtonRelease-1>", lambda event, v=view: self._on_roi_release(event, v))
        Label(right, textvariable=self.stats_text, justify="left", anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        Label(right, text="Live Results", anchor="w").grid(row=3, column=0, columnspan=2, sticky="ew")
        self.result_text = Text(right, height=8, font=("Menlo", 12), wrap="none", state="disabled")
        self.result_text.tag_configure("violation", foreground="red")
        self.result_text.grid(row=4, column=0, columnspan=2, sticky="ew")

    def _entry_row(self, parent: Frame, label: str, var) -> None:
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        Label(row, text=label, width=14, anchor="w").pack(side="left")
        Entry(row, textvariable=var, width=31).pack(side="right")

    def pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path.set(path)
        self._load_preview_frame(Path(path))

    def _load_preview_frame(self, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Error", f"Failed to read first frame: {video_path}")
            return
        self.status_text.set(f"Loaded preview: {video_path.name}. Draw ROI, then Start.")
        self.stats_text.set("Preview loaded")
        self._show_frames(frame, frame)

    def clear_roi(self) -> None:
        self.roi_rect = None
        self.roi_drag_start = None
        self.roi_drag_current = None
        self.status_text.set("ROI cleared")
        self._refresh_last_frame()

    def _display_to_frame(self, x: int, y: int, view: str) -> tuple[int, int] | None:
        meta = self.display_meta[view]
        image_w, image_h = meta["image_size"]
        frame_h, frame_w = meta["frame_shape"]
        off_x, off_y = meta["offset"]
        if image_w <= 0 or image_h <= 0 or frame_w <= 0 or frame_h <= 0:
            return None
        if not (off_x <= x <= off_x + image_w and off_y <= y <= off_y + image_h):
            return None
        fx = int((x - off_x) * frame_w / image_w)
        fy = int((y - off_y) * frame_h / image_h)
        return max(0, min(frame_w - 1, fx)), max(0, min(frame_h - 1, fy))

    def _on_roi_press(self, event, view: str) -> None:
        self.active_roi_view = view
        point = self._display_to_frame(event.x, event.y, view)
        if point is not None:
            self.roi_drag_start = point
            self.roi_drag_current = point

    def _on_roi_drag(self, event, view: str) -> None:
        if self.roi_drag_start is None:
            return
        point = self._display_to_frame(event.x, event.y, view)
        if point is not None:
            self.roi_drag_current = point
            self._refresh_last_frame()

    def _on_roi_release(self, event, view: str) -> None:
        if self.roi_drag_start is None:
            return
        point = self._display_to_frame(event.x, event.y, view)
        if point is None:
            self.roi_drag_start = None
            self.roi_drag_current = None
            return
        x1, y1 = self.roi_drag_start
        x2, y2 = point
        self.roi_drag_start = None
        self.roi_drag_current = None
        if abs(x2 - x1) >= 10 and abs(y2 - y1) >= 10:
            self.roi_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.status_text.set(f"ROI set: {self.roi_rect}")
        self._refresh_last_frame()

    def _refresh_last_frame(self) -> None:
        if self.last_wjh_frame is not None and self.last_yolo_frame is not None:
            self._show_frames(self.last_wjh_frame.copy(), self.last_yolo_frame.copy())

    def _center_in_roi(self, det: Detection) -> bool:
        if self.roi_rect is None:
            return True
        x1, y1, x2, y2 = self.roi_rect
        cx, cy = det.center
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _match_cross_targets(self, wjh_items, yolo_items):
        candidates = []
        for w_idx, w_item in enumerate(wjh_items):
            if not w_item["in_roi"]:
                continue
            for y_idx, y_item in enumerate(yolo_items):
                if not y_item["in_roi"]:
                    continue
                score = bbox_iou(w_item["det"].bbox, y_item["det"].bbox)
                if score >= self.cross_iou_thresh.get():
                    candidates.append((score, w_idx, y_idx))
        candidates.sort(reverse=True)

        matched = []
        used_w: set[int] = set()
        used_y: set[int] = set()
        for score, w_idx, y_idx in candidates:
            if w_idx in used_w or y_idx in used_y:
                continue
            used_w.add(w_idx)
            used_y.add(y_idx)
            matched.append((wjh_items[w_idx], yolo_items[y_idx], score))
        return matched

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Processing is already running.")
            return
        video = Path(self.video_path.get().strip())
        if not video.exists():
            messagebox.showwarning("No video", "Add a video first.")
            return
        self.result_rows = []
        self._replace_live_rows([])
        self.stop_event.clear()
        self.pause_event.clear()
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()

    def _run_worker(self) -> None:
        try:
            self._process_video(Path(self.video_path.get().strip()))
        except Exception as exc:
            self._put_ui({"type": "error", "text": str(exc)})

    def _process_video(self, video_path: Path) -> None:
        wjh_model = YOLO(self.wjh_model_path.get())
        yolo_model = YOLO(self.yolo_model_path.get())
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_dir = video_path.parent
        stem = f"{video_path.stem}-cross-validate-ljt"
        csv_path = out_dir / f"{stem}_frames.csv"
        target_csv_path = out_dir / f"{stem}_targets.csv"
        video_out_path = out_dir / f"{stem}.mp4"
        writer = None
        if self.save_video.get():
            writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, self.frame_stride.get()), (width * 2, height))

        rows: List[Dict[str, object]] = []
        target_rows: List[Dict[str, object]] = []
        frame_index = -1
        processed = 0
        final_frames = 0
        wjh_tracker = IouIdTracker(iou_thresh=0.25, max_missed=10, prefix="W")
        yolo_tracker = IouIdTracker(iou_thresh=0.25, max_missed=10, prefix="Y")
        target_states: Dict[str, Dict[str, object]] = {}
        live_violation_states: Dict[str, Dict[str, object]] = {}
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

            sync_cuda_if_available()
            wjh_result = wjh_model.predict(frame, classes=WJH_CLASSES, conf=self.conf.get(), iou=self.iou.get(), imgsz=self.imgsz.get(), verbose=False)[0]
            yolo_result = yolo_model.predict(frame, classes=YOLOV8N_CLASSES, conf=self.conf.get(), iou=self.iou.get(), imgsz=self.imgsz.get(), verbose=False)[0]
            sync_cuda_if_available()

            wjh_dets = result_to_detections(wjh_result, min_area=self.wjh_min_area.get())
            yolo_dets = result_to_detections(yolo_result, min_area=self.yolo_min_area.get())

            riders_all = [det for det in wjh_dets if det.class_id == RIDER_CLASS_ID]
            wjh_ids = wjh_tracker.update(riders_all)
            riders = [det for det in riders_all if self._center_in_roi(det)]
            heads = [det for det in wjh_dets if det.class_id == HEAD_CLASS_ID and det.confidence >= self.head_conf.get()]
            helmets = [det for det in wjh_dets if det.class_id == HELMET_CLASS_ID and det.confidence >= self.helmet_conf.get()]
            wjh_items = []
            for rider_idx, rider in enumerate(riders_all):
                people, stats = build_person_evidence_for_rider(rider, heads, helmets)
                in_roi = self._center_in_roi(rider)
                overload = in_roi and int(stats["person_evidence_count"]) >= 2
                helmet_status = "NO_HELMET" if int(stats["unpaired_head_count"]) > 0 else "HELMETED" if int(stats["helmet_count"]) > 0 else "UNKNOWN"
                wjh_items.append(
                    {
                        "det": rider,
                        "id": wjh_ids[rider_idx],
                        "people": people,
                        "stats": stats,
                        "overload": overload,
                        "helmet_status": helmet_status,
                        "in_roi": in_roi,
                    }
                )
            wjh_overloads = [item for item in wjh_items if item["overload"]]

            people = [det for det in yolo_dets if det.class_id == PERSON_CLASS_ID]
            motorcycles_all = [det for det in yolo_dets if det.class_id == MOTORCYCLE_CLASS_ID]
            yolo_ids = yolo_tracker.update(motorcycles_all)
            motorcycles = [det for det in motorcycles_all if self._center_in_roi(det)]
            grouped_people, _scores = match_people_to_vehicles(people, motorcycles, self.yolo_match_thresh.get())
            yolo_items = []
            for moto_all_idx, motorcycle in enumerate(motorcycles_all):
                in_roi = self._center_in_roi(motorcycle)
                roi_idx = motorcycles.index(motorcycle) if motorcycle in motorcycles else None
                matched = grouped_people[roi_idx] if roi_idx is not None else []
                yolo_items.append(
                    {
                        "det": motorcycle,
                        "id": yolo_ids[moto_all_idx],
                        "people": matched,
                        "overload": in_roi and len(matched) >= 2,
                        "in_roi": in_roi,
                    }
                )
            yolo_overloads = [item for item in yolo_items if item["overload"]]

            target_matches = self._match_cross_targets(wjh_items, yolo_items)
            final_items = []
            for w_item, y_item, cross_iou in target_matches:
                target_id = f"{w_item['id']}|{y_item['id']}"
                final_now = bool(w_item["overload"] and y_item["overload"])
                state = target_states.setdefault(
                    target_id,
                    {
                        "ever_overload": False,
                        "last_frame": 0,
                        "helmet_status": "UNKNOWN",
                        "wjh_id": w_item["id"],
                        "yolo_id": y_item["id"],
                    },
                )
                if final_now:
                    state["ever_overload"] = True
                if w_item["helmet_status"] != "UNKNOWN":
                    state["helmet_status"] = w_item["helmet_status"]
                state["last_frame"] = frame_index
                final_items.append((target_id, w_item, y_item, cross_iou, final_now, state))
                target_rows.append(
                    {
                        "frame": frame_index,
                        "target_id": target_id,
                        "wjh_id": w_item["id"],
                        "yolo_id": y_item["id"],
                        "cross_iou": f"{cross_iou:.4f}",
                        "wjh_overload": int(w_item["overload"]),
                        "yolov8n_overload": int(y_item["overload"]),
                        "final_overload": int(final_now),
                        "ever_final_overload": int(bool(state["ever_overload"])),
                        "helmet_status": state["helmet_status"],
                        "wjh_person_evidence_count": int(w_item["stats"]["person_evidence_count"]),
                        "wjh_no_helmet_count": int(w_item["stats"]["unpaired_head_count"]),
                        "yolov8n_person_count": len(y_item["people"]),
                    }
                )

            for target_id, w_item, y_item, cross_iou, final_now, state in final_items:
                if final_now:
                    self._record_live_violation(
                        live_violation_states,
                        frame_index,
                        violation_type="OVERLOAD",
                        target_id=target_id,
                        wjh_id=w_item["id"],
                        yolo_id=y_item["id"],
                        detail=(
                            f"cross_iou={cross_iou:.2f} wjh_count={int(w_item['stats']['person_evidence_count'])} "
                            f"yolo_count={len(y_item['people'])} helmet={state['helmet_status']}"
                        ),
                    )
                if state["helmet_status"] == "NO_HELMET":
                    self._record_live_violation(
                        live_violation_states,
                        frame_index,
                        violation_type="NO_HELMET",
                        target_id=target_id,
                        wjh_id=w_item["id"],
                        yolo_id=y_item["id"],
                        detail=f"cross_iou={cross_iou:.2f} no_helmet_heads={int(w_item['stats']['unpaired_head_count'])}",
                    )

            final_overload = any(item[4] for item in final_items)
            final_frames += int(final_overload)
            wjh_annotated = self._draw_wjh_frame(frame, riders_all, heads, helmets, wjh_items, final_items, frame_index)
            yolo_annotated = self._draw_yolo_frame(frame, people, motorcycles_all, yolo_items, final_items, frame_index)
            if writer is not None:
                writer.write(cv2.hconcat([wjh_annotated, yolo_annotated]))

            rows.append(
                {
                    "frame": frame_index,
                    "wjh_overload_count": len(wjh_overloads),
                    "yolov8n_overload_count": len(yolo_overloads),
                    "matched_target_count": len(final_items),
                    "final_overload": int(final_overload),
                    "final_target_ids": " ".join(item[0] for item in final_items if item[4]),
                    "helmet_statuses": " ".join(f"{item[0]}:{item[5]['helmet_status']}" for item in final_items),
                    "roi": " ".join(str(v) for v in self.roi_rect) if self.roi_rect else "",
                }
            )

            processed += 1
            wall_fps = processed / max(1e-6, time.perf_counter() - start_wall)
            self._put_ui(
                {
                    "type": "frame",
                    "wjh_frame": wjh_annotated,
                    "yolo_frame": yolo_annotated,
                    "status": f"Processing: {video_path.name}",
                    "stats": (
                        f"frame={frame_index}/{total_frames} wall_fps={wall_fps:.1f} "
                        f"targets={len(final_items)} wjh_overload={len(wjh_overloads)} yolov8n_overload={len(yolo_overloads)} "
                        f"final_frames={final_frames}"
                    ),
                    "live_rows": self._format_live_violation_rows(live_violation_states),
                }
            )

        cap.release()
        if writer is not None:
            writer.release()
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "wjh_overload_count",
                    "yolov8n_overload_count",
                    "matched_target_count",
                    "final_overload",
                    "final_target_ids",
                    "helmet_statuses",
                    "roi",
                ],
            )
            writer_csv.writeheader()
            writer_csv.writerows(rows)
        with target_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "target_id",
                    "wjh_id",
                    "yolo_id",
                    "cross_iou",
                    "wjh_overload",
                    "yolov8n_overload",
                    "final_overload",
                    "ever_final_overload",
                    "helmet_status",
                    "wjh_person_evidence_count",
                    "wjh_no_helmet_count",
                    "yolov8n_person_count",
                ],
            )
            writer_csv.writeheader()
            writer_csv.writerows(target_rows)
        self._put_ui({"type": "status", "text": f"Saved: {csv_path} and {target_csv_path}"})

    def _draw_wjh_frame(self, frame, riders, heads, helmets, wjh_items, final_items, frame_index: int):
        vis = frame.copy()
        overload_ids = {id(item["det"]) for item in wjh_items if item["overload"]}
        final_by_wjh_id = {item[1]["id"]: item for item in final_items}
        for rider in riders:
            x1, y1, x2, y2 = [int(v) for v in rider.bbox]
            color = (0, 140, 255) if id(rider) in overload_ids else (255, 160, 0)
            thick = 2 if id(rider) in overload_ids else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            w_item = next((item for item in wjh_items if item["det"] is rider), None)
            label = f"{w_item['id']} rider {rider.confidence:.2f}" if w_item else f"rider {rider.confidence:.2f}"
            if w_item and w_item["id"] in final_by_wjh_id:
                final_item = final_by_wjh_id[w_item["id"]]
                label += f" {final_item[0]} {final_item[5]['helmet_status']}"
            cv2.putText(vis, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for head in heads:
            x1, y1, x2, y2 = [int(v) for v in head.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 1)
            cv2.putText(vis, f"head {head.confidence:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
        for helmet in helmets:
            x1, y1, x2, y2 = [int(v) for v in helmet.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 220, 40), 1)
            cv2.putText(vis, f"helmet {helmet.confidence:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 220, 40), 1)
        for item in wjh_items:
            if not item["overload"]:
                continue
            rider = item["det"]
            people = item["people"]
            stats = item["stats"]
            x1, y1, x2, y2 = [int(v) for v in rider.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(vis, f"wjh persons={stats['person_evidence_count']}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
            for item in people:
                ex1, ey1, ex2, ey2 = [int(v) for v in item.bbox]
                cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (0, 220, 220), 1)

        vis = self._draw_roi_overlay(vis)
        final_now = any(item[4] for item in final_items)
        color = (0, 0, 255) if final_now else (40, 180, 40)
        label = "FINAL OVERLOAD" if final_now else "final normal"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(vis, f"wjh frame={frame_index} {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return vis

    def _draw_yolo_frame(self, frame, people, motorcycles, yolo_items, final_items, frame_index: int):
        vis = frame.copy()
        overload_ids = {id(item["det"]) for item in yolo_items if item["overload"]}
        final_by_yolo_id = {item[2]["id"]: item for item in final_items}
        for person in people:
            x1, y1, x2, y2 = [int(v) for v in person.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 1)
            cv2.putText(vis, f"person {person.confidence:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
        for motorcycle in motorcycles:
            x1, y1, x2, y2 = [int(v) for v in motorcycle.bbox]
            color = (255, 0, 255) if id(motorcycle) in overload_ids else (255, 160, 0)
            thick = 2 if id(motorcycle) in overload_ids else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            y_item = next((item for item in yolo_items if item["det"] is motorcycle), None)
            label = f"{y_item['id']} motorcycle {motorcycle.confidence:.2f}" if y_item else f"motorcycle {motorcycle.confidence:.2f}"
            if y_item and y_item["id"] in final_by_yolo_id:
                final_item = final_by_yolo_id[y_item["id"]]
                label += f" {final_item[0]} {final_item[5]['helmet_status']}"
            cv2.putText(vis, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for item in yolo_items:
            if not item["overload"]:
                continue
            motorcycle = item["det"]
            matched_people = item["people"]
            x1, y1, x2, y2 = [int(v) for v in motorcycle.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(vis, f"yolo persons={len(matched_people)}", (x1, min(vis.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            for person in matched_people:
                px1, py1, px2, py2 = [int(v) for v in person.bbox]
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 220, 0), 1)
        vis = self._draw_roi_overlay(vis)
        final_now = any(item[4] for item in final_items)
        color = (0, 0, 255) if final_now else (40, 180, 40)
        label = "FINAL OVERLOAD" if final_now else "final normal"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(vis, f"yolov8n frame={frame_index} {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return vis

    def _draw_roi_overlay(self, frame):
        vis = frame.copy()
        if self.roi_rect is not None:
            x1, y1, x2, y2 = self.roi_rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(vis, "valid ROI", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        if self.roi_drag_start and self.roi_drag_current:
            x1, y1 = self.roi_drag_start
            x2, y2 = self.roi_drag_current
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
        return vis

    def _show_single_frame(self, view: str, frame, label: Label) -> None:
        frame = self._draw_roi_overlay(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        label_w = max(420, label.winfo_width())
        label_h = max(300, label.winfo_height())
        img.thumbnail((label_w, label_h))
        self.display_meta[view] = {
            "image_size": img.size,
            "frame_shape": frame.shape[:2],
            "offset": ((label_w - img.size[0]) // 2, (label_h - img.size[1]) // 2),
        }
        photo = ImageTk.PhotoImage(img)
        if view == "wjh":
            self.wjh_photo = photo
            self.wjh_label.configure(image=self.wjh_photo)
        else:
            self.yolo_photo = photo
            self.yolo_label.configure(image=self.yolo_photo)

    def _show_frames(self, wjh_frame, yolo_frame) -> None:
        self.last_wjh_frame = wjh_frame.copy()
        self.last_yolo_frame = yolo_frame.copy()
        self._show_single_frame("wjh", wjh_frame, self.wjh_label)
        self._show_single_frame("yolo", yolo_frame, self.yolo_label)

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
                    self._replace_live_rows([str(row) for row in item.get("live_rows", [])])
                    self._show_frames(item["wjh_frame"], item["yolo_frame"])
                elif item["type"] == "status":
                    self.status_text.set(str(item["text"]))
                elif item["type"] == "error":
                    self.status_text.set("Error")
                    messagebox.showerror("Processing error", str(item["text"]))
        except queue.Empty:
            pass
        self.root.after(30, self._poll_ui_queue)

    def _record_live_violation(
        self,
        states: Dict[str, Dict[str, object]],
        frame_index: int,
        violation_type: str,
        target_id: str,
        wjh_id: str,
        yolo_id: str,
        detail: str,
    ) -> None:
        key = f"{violation_type}:{target_id}"
        state = states.setdefault(
            key,
            {
                "violation_type": violation_type,
                "target_id": target_id,
                "wjh_id": wjh_id,
                "yolo_id": yolo_id,
                "first_frame": frame_index,
                "last_frame": frame_index,
                "hits": 0,
                "detail": detail,
            },
        )
        state["last_frame"] = frame_index
        state["hits"] = int(state["hits"]) + 1
        state["detail"] = detail

    def _format_live_violation_rows(self, states: Dict[str, Dict[str, object]]) -> List[str]:
        grouped: Dict[str, Dict[str, object]] = {}
        for state in states.values():
            target_id = str(state["target_id"])
            if "NA" in target_id or state["yolo_id"] == "NA":
                continue
            target_state = grouped.setdefault(
                target_id,
                {
                    "target_id": target_id,
                    "wjh_id": state["wjh_id"],
                    "yolo_id": state["yolo_id"],
                    "first_frame": int(state["first_frame"]),
                    "last_frame": int(state["last_frame"]),
                    "violations": [],
                },
            )
            target_state["first_frame"] = min(int(target_state["first_frame"]), int(state["first_frame"]))
            target_state["last_frame"] = max(int(target_state["last_frame"]), int(state["last_frame"]))
            target_state["violations"].append(
                f"{state['violation_type']}(frames={state['first_frame']}-{state['last_frame']} hits={state['hits']})"
            )
            target_state.setdefault("details", []).append(f"{state['violation_type']}:{state['detail']}")

        sorted_states = sorted(grouped.values(), key=lambda item: int(item["last_frame"]), reverse=True)
        rows = []
        for state in sorted_states[:100]:
            rows.append(
                "; ".join(str(item) for item in state["violations"])
                + f" target={state['target_id']} "
                f"wjh_id={state['wjh_id']} yolo_id={state['yolo_id']} "
                f"frames={state['first_frame']}-{state['last_frame']} "
                f"details={' | '.join(str(item) for item in state.get('details', []))}"
            )
        return rows

    def _replace_live_rows(self, rows: List[str]) -> None:
        self.result_rows = rows
        self.result_rows = self.result_rows[:100]
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        for row in self.result_rows:
            line_start = self.result_text.index("end-1c")
            self.result_text.insert("end", row + "\n")
            for token in ("OVERLOAD", "NO_HELMET"):
                search_from = line_start
                while True:
                    match_start = self.result_text.search(token, search_from, stopindex="end")
                    if not match_start:
                        break
                    match_end = f"{match_start}+{len(token)}c"
                    self.result_text.tag_add("violation", match_start, match_end)
                    search_from = match_end
        self.result_text.configure(state="disabled")


def main() -> None:
    root = Tk()
    CrossValidateGuiLjt(root)
    root.mainloop()


if __name__ == "__main__":
    main()
