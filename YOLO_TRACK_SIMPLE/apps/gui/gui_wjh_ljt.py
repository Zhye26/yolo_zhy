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
    Canvas,
    Checkbutton,
    DoubleVar,
    Entry,
    Frame,
    IntVar,
    Label,
    Listbox,
    Scrollbar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from typing import Dict, List, Tuple

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from methods.association_ljt import AssociationByteTracker, AssociationMotTracker
from core.pipeline.pipeline_ljt import Detection, FrameVehicleResult, expand_box, overlap_ratio, point_in_box, result_to_detections, sync_cuda_if_available


RIDER_CLASS_ID = 0
HEAD_CLASS_ID = 1
HELMET_CLASS_ID = 2
TARGET_CLASSES = [RIDER_CLASS_ID, HEAD_CLASS_ID, HELMET_CLASS_ID]
CLASS_NAMES = {
    RIDER_CLASS_ID: "rider",
    HEAD_CLASS_ID: "head",
    HELMET_CLASS_ID: "helmet",
}


class SimpleIouObjectTracker:
    def __init__(self, iou_thresh: float = 0.25, max_missed: int = 10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    def update(self, det_array, frame):
        del frame
        detections = []
        for row in det_array:
            detections.append(
                {
                    "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                    "conf": float(row[4]),
                    "cls": int(row[5]),
                }
            )

        matches: Dict[int, int] = {}
        used_tracks: set[int] = set()
        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            if int(track["missed"]) > self.max_missed:
                continue
            for det_idx, det in enumerate(detections):
                score = self._iou(track["bbox"], det["bbox"])
                if score >= self.iou_thresh:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(reverse=True)

        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in matches:
                continue
            used_tracks.add(track_id)
            matches[det_idx] = track_id

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track["missed"] = int(track["missed"]) + 1

        for det_idx, det in enumerate(detections):
            track_id = matches.get(det_idx)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                matches[det_idx] = track_id
            self.tracks[track_id] = {"bbox": det["bbox"], "missed": 0, "cls": det["cls"], "conf": det["conf"]}

        rows = []
        for det_idx, track_id in matches.items():
            det = detections[det_idx]
            rows.append([*det["bbox"], track_id, det["conf"], det["cls"]])
        return rows

    @staticmethod
    def _iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0


def _center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def occupant_rider_match_score(occupant: Detection, rider: Detection) -> float:
    cx, cy = _center(occupant.bbox)
    rx1, ry1, rx2, ry2 = rider.bbox
    rw = max(1.0, rx2 - rx1)
    rh = max(1.0, ry2 - ry1)
    support = expand_box(rider.bbox, x_ratio=0.18, top_ratio=0.12, bottom_ratio=0.05)
    if not point_in_box(cx, cy, support):
        return 0.0

    vertical = (cy - ry1) / rh
    horizontal_gap = abs(cx - (rx1 + rx2) / 2.0) / rw
    score = 0.55
    score += min(overlap_ratio(occupant.bbox, rider.bbox) * 0.55, 0.30)
    score += max(0.0, 0.20 - horizontal_gap * 0.20)
    if 0.02 <= vertical <= 0.72:
        score += 0.18
    elif vertical > 0.88:
        score -= 0.35
    return max(0.0, score)


def build_head_helmet_evidence(heads: List[Detection], helmets: List[Detection]) -> List[Detection]:
    evidence: List[Detection] = []
    for head in heads:
        occ = Detection(class_id=HEAD_CLASS_ID, confidence=head.confidence, bbox=head.bbox)
        occ.has_helmet = False
        occ.source_class_name = "head"
        evidence.append(occ)
    for helmet in helmets:
        occ = Detection(class_id=HELMET_CLASS_ID, confidence=helmet.confidence, bbox=helmet.bbox)
        occ.has_helmet = True
        occ.helmet_confidence = helmet.confidence
        occ.source_class_name = "helmet"
        evidence.append(occ)
    return evidence


def _copy_evidence(det: Detection, class_id: int, source_class_name: str, has_helmet: bool) -> Detection:
    copied = Detection(class_id=class_id, confidence=det.confidence, bbox=det.bbox)
    copied.has_helmet = has_helmet
    copied.source_class_name = source_class_name
    copied.helmet_confidence = det.confidence if has_helmet else 0.0
    return copied


def _evidence_inside_rider(evidence: Detection, rider: Detection, min_overlap: float = 0.80) -> bool:
    cx, cy = _center(evidence.bbox)
    return point_in_box(cx, cy, rider.bbox) and overlap_ratio(evidence.bbox, rider.bbox) >= min_overlap


def build_person_evidence_for_rider(
    rider: Detection,
    heads: List[Detection],
    helmets: List[Detection],
    min_rider_overlap: float = 0.80,
    min_pair_overlap: float = 0.30,
) -> Tuple[List[Detection], Dict[str, int]]:
    inside_heads = [(idx, head) for idx, head in enumerate(heads) if _evidence_inside_rider(head, rider, min_rider_overlap)]
    inside_helmets = [
        (idx, helmet) for idx, helmet in enumerate(helmets) if _evidence_inside_rider(helmet, rider, min_rider_overlap)
    ]

    candidates: List[Tuple[float, int, int]] = []
    for head_idx, head in inside_heads:
        for helmet_idx, helmet in inside_helmets:
            pair_overlap = overlap_ratio(head.bbox, helmet.bbox)
            if pair_overlap >= min_pair_overlap:
                candidates.append((pair_overlap, head_idx, helmet_idx))
    candidates.sort(reverse=True)

    used_heads: set[int] = set()
    used_helmets: set[int] = set()
    person_evidence: List[Detection] = []
    for _, head_idx, helmet_idx in candidates:
        if head_idx in used_heads or helmet_idx in used_helmets:
            continue
        head = heads[head_idx]
        helmet = helmets[helmet_idx]
        merged = Detection(
            class_id=HELMET_CLASS_ID,
            confidence=max(head.confidence, helmet.confidence),
            bbox=helmet.bbox,
        )
        merged.has_helmet = True
        merged.source_class_name = "head+helmet"
        merged.helmet_confidence = helmet.confidence
        person_evidence.append(merged)
        used_heads.add(head_idx)
        used_helmets.add(helmet_idx)

    for head_idx, head in inside_heads:
        if head_idx not in used_heads:
            person_evidence.append(_copy_evidence(head, HEAD_CLASS_ID, "head", False))
    for helmet_idx, helmet in inside_helmets:
        if helmet_idx not in used_helmets:
            person_evidence.append(_copy_evidence(helmet, HELMET_CLASS_ID, "helmet", True))

    stats = {
        "head_count": len(inside_heads),
        "helmet_count": len(inside_helmets),
        "paired_head_helmet_count": len(used_heads),
        "unpaired_head_count": len(inside_heads) - len(used_heads),
        "unpaired_helmet_count": len(inside_helmets) - len(used_helmets),
        "person_evidence_count": len(person_evidence),
    }
    return person_evidence, stats


def merge_head_helmet_detections(heads: List[Detection], helmets: List[Detection]) -> List[Detection]:
    occupants: List[Detection] = []
    used_helmets: set[int] = set()

    for head in heads:
        best_idx = None
        best_score = 0.0
        hx, hy = _center(head.bbox)
        for idx, helmet in enumerate(helmets):
            if idx in used_helmets:
                continue
            hcx, hcy = _center(helmet.bbox)
            expanded_head = expand_box(head.bbox, x_ratio=0.55, top_ratio=0.55, bottom_ratio=0.55)
            score = overlap_ratio(head.bbox, helmet.bbox)
            if point_in_box(hcx, hcy, expanded_head) or point_in_box(hx, hy, expand_box(helmet.bbox, 0.35, 0.35, 0.35)):
                score = max(score, 0.35)
            if score > best_score:
                best_idx = idx
                best_score = score

        occ = Detection(class_id=HEAD_CLASS_ID, confidence=head.confidence, bbox=head.bbox)
        occ.has_helmet = best_idx is not None and best_score >= 0.20
        occ.helmet_confidence = helmets[best_idx].confidence if occ.has_helmet else 0.0
        occ.source_class_name = "head+helmet" if occ.has_helmet else "head"
        if best_idx is not None and best_score >= 0.20:
            used_helmets.add(best_idx)
        occupants.append(occ)

    for idx, helmet in enumerate(helmets):
        if idx in used_helmets:
            continue
        occ = Detection(class_id=HELMET_CLASS_ID, confidence=helmet.confidence, bbox=helmet.bbox)
        occ.has_helmet = True
        occ.helmet_confidence = helmet.confidence
        occ.source_class_name = "helmet"
        occupants.append(occ)

    return occupants


def draw_wjh_frame(
    frame,
    occupants: List[Detection],
    rider_results: List[FrameVehicleResult],
    frame_index: int,
    timestamp_sec: float,
    fps_text: str,
):
    vis = frame.copy()
    matched_ids = {id(occ) for item in rider_results for occ in item.matched_people}

    for occ in occupants:
        x1, y1, x2, y2 = [int(v) for v in occ.bbox]
        helmeted = bool(getattr(occ, "has_helmet", False))
        color = (60, 210, 60) if helmeted else (0, 180, 255)
        if id(occ) not in matched_ids:
            color = (130, 130, 130)
        label = "helmet" if helmeted else "no_helmet"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, f"{label} {occ.confidence:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    for item in rider_results:
        det = item.detection
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        helmeted = sum(1 for occ in item.matched_people if occ.class_id == HELMET_CLASS_ID)
        head_count = sum(1 for occ in item.matched_people if occ.class_id == HEAD_CLASS_ID)
        no_helmet = sum(1 for occ in item.matched_people if occ.class_id == HEAD_CLASS_ID and not bool(getattr(occ, "has_helmet", False)))
        color = (0, 0, 255) if item.confirmed_overload else (0, 165, 255) if item.raw_overload else (255, 160, 0)
        label = "OVERLOAD" if item.confirmed_overload else "candidate" if item.raw_overload else "rider"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"#{item.track_id} {label} persons={len(item.matched_people)} head={head_count} helmet={helmeted} no_helmet={no_helmet}",
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        rcx = int((x1 + x2) / 2)
        rcy = int((y1 + y2) / 2)
        for occ in item.matched_people:
            ocx, ocy = _center(occ.bbox)
            cv2.line(vis, (rcx, rcy), (int(ocx), int(ocy)), color, 2)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(vis, f"wjh-association frame={frame_index} t={timestamp_sec:.2f}s {fps_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


class WjhAssociationGuiLjt:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("wjh.pt Rider/Head/Helmet Association - ljt")
        self.root.geometry("1180x820")

        self.video_paths: List[Path] = []
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.ui_queue: queue.Queue = queue.Queue(maxsize=3)
        self.model = None
        self.current_photo = None
        self.roi_rect: tuple[int, int, int, int] | None = None
        self.roi_drag_start: tuple[int, int] | None = None
        self.roi_drag_current: tuple[int, int] | None = None
        self.display_image_size: tuple[int, int] = (0, 0)
        self.display_offset: tuple[int, int] = (0, 0)
        self.display_frame_shape: tuple[int, int] = (0, 0)
        self.last_display_frame = None

        project_root = Path(__file__).resolve().parents[2]
        default_model = project_root / "weights" / "wjh.pt"
        self.model_path = StringVar(value=str(default_model))
        self.detect_classes = StringVar(value="0,1,2")
        self.conf = DoubleVar(value=0.25)
        self.head_conf = DoubleVar(value=0.25)
        self.helmet_conf = DoubleVar(value=0.45)
        self.iou = DoubleVar(value=0.45)
        self.imgsz = IntVar(value=640)
        self.match_thresh = DoubleVar(value=0.55)
        self.confirm_frames = IntVar(value=2)
        self.max_missed = IntVar(value=6)
        self.byte_track_thresh = DoubleVar(value=0.25)
        self.byte_match_thresh = DoubleVar(value=0.8)
        self.byte_track_buffer = IntVar(value=30)
        self.association_min_hits = IntVar(value=2)
        self.association_lock_frames = IntVar(value=20)
        self.association_unbind_frames = IntVar(value=15)
        self.association_switch_margin = DoubleVar(value=0.25)
        self.min_area = DoubleVar(value=12.0)
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
        left.bind("<Configure>", lambda _event=None: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(left_window, width=event.width))

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

        roi_actions = Frame(left)
        roi_actions.pack(fill="x", pady=(0, 6))
        Button(roi_actions, text="Clear ROI", command=self.clear_roi, width=12).pack(side="left", padx=(0, 5))
        Label(roi_actions, text="Drag on video to set ROI", anchor="w").pack(side="left")

        Label(left, textvariable=self.status_text, wraplength=360, justify="left").pack(fill="x", pady=(0, 8), anchor="w")
        self.video_list = Listbox(left, width=46, height=8)
        self.video_list.pack(pady=8)

        for label, var in (
            ("Model", self.model_path),
            ("detect_classes", self.detect_classes),
            ("conf", self.conf),
            ("head_conf", self.head_conf),
            ("helmet_conf", self.helmet_conf),
            ("iou", self.iou),
            ("imgsz", self.imgsz),
            ("match_thresh", self.match_thresh),
            ("confirm_frames", self.confirm_frames),
            ("max_missed", self.max_missed),
            ("byte_track_thresh", self.byte_track_thresh),
            ("byte_match_thresh", self.byte_match_thresh),
            ("byte_track_buffer", self.byte_track_buffer),
            ("assoc_min_hits", self.association_min_hits),
            ("assoc_lock", self.association_lock_frames),
            ("assoc_unbind", self.association_unbind_frames),
            ("assoc_switch", self.association_switch_margin),
            ("min_area", self.min_area),
            ("frame_stride", self.frame_stride),
        ):
            self._labeled_entry(left, label, var)
        Checkbutton(left, text="Save annotated videos", variable=self.save_video).pack(anchor="w", pady=6)

        self.video_label = Label(right, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<ButtonPress-1>", self._on_roi_press)
        self.video_label.bind("<B1-Motion>", self._on_roi_drag)
        self.video_label.bind("<ButtonRelease-1>", self._on_roi_release)
        Label(right, textvariable=self.stats_text, justify="left", anchor="w").grid(row=1, column=0, sticky="ew", pady=(8, 4))
        Label(right, text="Track Results", anchor="w").grid(row=2, column=0, sticky="ew")
        self.track_result_list = Listbox(right, height=8, font=("Menlo", 12))
        self.track_result_list.grid(row=3, column=0, sticky="ew")

    def _labeled_entry(self, parent: Frame, label: str, variable) -> None:
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        Label(row, text=label, width=14, anchor="w").pack(side="left")
        Entry(row, textvariable=variable, width=24).pack(side="right")

    def add_videos(self) -> None:
        paths = filedialog.askopenfilenames(title="Select videos", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
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

    def clear_roi(self) -> None:
        self.roi_rect = None
        self.roi_drag_start = None
        self.roi_drag_current = None
        self.status_text.set("ROI cleared")
        self._refresh_last_frame()

    def _display_to_frame(self, x: int, y: int) -> tuple[int, int] | None:
        image_w, image_h = self.display_image_size
        frame_h, frame_w = self.display_frame_shape
        off_x, off_y = self.display_offset
        if image_w <= 0 or image_h <= 0 or frame_w <= 0 or frame_h <= 0:
            return None
        if not (off_x <= x <= off_x + image_w and off_y <= y <= off_y + image_h):
            return None
        fx = int((x - off_x) * frame_w / image_w)
        fy = int((y - off_y) * frame_h / image_h)
        return max(0, min(frame_w - 1, fx)), max(0, min(frame_h - 1, fy))

    def _on_roi_press(self, event) -> None:
        point = self._display_to_frame(event.x, event.y)
        if point is None:
            return
        self.roi_drag_start = point
        self.roi_drag_current = point

    def _on_roi_drag(self, event) -> None:
        if self.roi_drag_start is None:
            return
        point = self._display_to_frame(event.x, event.y)
        if point is None:
            return
        self.roi_drag_current = point
        self._refresh_last_frame()

    def _on_roi_release(self, event) -> None:
        if self.roi_drag_start is None:
            return
        point = self._display_to_frame(event.x, event.y)
        if point is None:
            self.roi_drag_start = None
            self.roi_drag_current = None
            return
        x1, y1 = self.roi_drag_start
        x2, y2 = point
        self.roi_drag_start = None
        self.roi_drag_current = None
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.status_text.set("ROI ignored: drag a larger rectangle")
            self._refresh_last_frame()
            return
        self.roi_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.status_text.set(f"ROI set: {self.roi_rect}")
        self._refresh_last_frame()

    def _refresh_last_frame(self) -> None:
        if self.last_display_frame is not None:
            self._show_frame(self.last_display_frame.copy())

    def _rider_in_valid_region(self, rider: Detection) -> bool:
        if self.roi_rect is None:
            return True
        x1, y1, x2, y2 = self.roi_rect
        cx, cy = rider.center
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _draw_roi_overlay(self, frame):
        vis = frame.copy()
        if self.roi_rect is not None:
            x1, y1, x2, y2 = self.roi_rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(vis, "valid ROI", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        if self.roi_drag_start is not None and self.roi_drag_current is not None:
            x1, y1 = self.roi_drag_start
            x2, y2 = self.roi_drag_current
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
        return vis

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
        self.pause_event.clear() if self.pause_event.is_set() else self.pause_event.set()
        self.status_text.set("Paused" if self.pause_event.is_set() else "Running")

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
            return TARGET_CLASSES
        return [int(item.strip()) for item in raw.split(",") if item.strip()]

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

    def _new_tracker(self):
        try:
            return AssociationByteTracker(
                track_thresh=self.byte_track_thresh.get(),
                match_thresh=self.byte_match_thresh.get(),
                track_buffer=self.byte_track_buffer.get(),
                confirm_frames=self.confirm_frames.get(),
                max_missed=self.max_missed.get(),
                association_min_hits=self.association_min_hits.get(),
                association_lock_frames=self.association_lock_frames.get(),
                association_unbind_frames=self.association_unbind_frames.get(),
                association_switch_margin=self.association_switch_margin.get(),
                match_score_fn=occupant_rider_match_score,
            )
        except RuntimeError:
            return AssociationMotTracker(
                person_tracker=SimpleIouObjectTracker(iou_thresh=0.20, max_missed=self.max_missed.get()),
                vehicle_tracker=SimpleIouObjectTracker(iou_thresh=0.20, max_missed=self.max_missed.get()),
                confirm_frames=self.confirm_frames.get(),
                max_missed=self.max_missed.get(),
                association_min_hits=self.association_min_hits.get(),
                association_lock_frames=self.association_lock_frames.get(),
                association_unbind_frames=self.association_unbind_frames.get(),
                association_switch_margin=self.association_switch_margin.get(),
                match_score_fn=occupant_rider_match_score,
            )

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
        tracker = self._new_tracker()

        output_dir = video_path.parent
        stem = f"{video_path.stem}-wjh-association-ljt"
        frame_csv_path = output_dir / f"{stem}_frames.csv"
        summary_csv_path = output_dir / f"{stem}_summary.csv"
        video_out_path = output_dir / f"{stem}.mp4"
        writer = None
        if self.save_video.get():
            writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps / max(1, self.frame_stride.get()), (frame_w, frame_h))

        rows: List[Dict[str, object]] = []
        track_summary: Dict[str, Dict[str, int | bool | str | float]] = {}
        overload_track_ids: set[str] = set()
        timing_ms: List[float] = []
        processed = 0
        frame_index = -1
        frames_with_raw = 0
        frames_with_confirmed = 0
        frames_with_no_helmet = 0
        total_occupants = 0
        total_riders = 0
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
            result = model.predict(frame, classes=detect_classes, conf=self.conf.get(), iou=self.iou.get(), imgsz=self.imgsz.get(), verbose=False)[0]
            sync_cuda_if_available()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timing_ms.append(elapsed_ms)

            detections = result_to_detections(result, min_area=self.min_area.get())
            riders = [det for det in detections if det.class_id == RIDER_CLASS_ID]
            heads = [
                det for det in detections
                if det.class_id == HEAD_CLASS_ID and det.confidence >= self.head_conf.get()
            ]
            helmets = [
                det for det in detections
                if det.class_id == HELMET_CLASS_ID and det.confidence >= self.helmet_conf.get()
            ]
            occupants = build_head_helmet_evidence(heads, helmets)
            total_occupants += len(occupants)
            total_riders += len(riders)

            track_matches, grouped_occupants, match_scores = tracker.update_scene(occupants, riders, frame, self.match_thresh.get())
            frame_results: List[FrameVehicleResult] = []
            has_raw = False
            has_confirmed = False
            has_no_helmet = False

            for rider_idx, rider in enumerate(riders):
                track_id = track_matches[rider_idx]
                track_key = str(track_id)
                in_valid_region = self._rider_in_valid_region(rider)
                matched, geometry_stats = build_person_evidence_for_rider(rider, heads, helmets)
                raw_overload_candidate = int(geometry_stats["person_evidence_count"]) >= 2
                raw_overload = raw_overload_candidate and in_valid_region
                if raw_overload and not track_key.startswith("U"):
                    overload_track_ids.add(track_key)
                confirmed = track_key in overload_track_ids
                overload_status = "CONFIRMED" if confirmed else "NORMAL"
                if str(track_id).startswith("U"):
                    raw_overload = False
                    confirmed = False
                    overload_status = "UNCERTAIN"
                head_count = int(geometry_stats["head_count"])
                helmeted_count = int(geometry_stats["helmet_count"])
                person_evidence_count = int(geometry_stats["person_evidence_count"])
                paired_head_helmet_count = int(geometry_stats["paired_head_helmet_count"])
                unpaired_head_count = int(geometry_stats["unpaired_head_count"])
                unpaired_helmet_count = int(geometry_stats["unpaired_helmet_count"])
                no_helmet_count = unpaired_head_count
                judgment_updated = in_valid_region and not str(track_id).startswith("U")
                has_raw = has_raw or (judgment_updated and raw_overload)
                has_confirmed = has_confirmed or (judgment_updated and confirmed)
                has_no_helmet = has_no_helmet or (judgment_updated and no_helmet_count > 0)
                if judgment_updated:
                    self._update_track_summary(track_summary, track_id, raw_overload, confirmed, person_evidence_count, helmeted_count, no_helmet_count, frame_index, overload_status)
                frame_results.append(
                    FrameVehicleResult(
                        track_id=track_id,
                        detection=rider,
                        matched_people=matched,
                        match_scores=[],
                        raw_overload=raw_overload_candidate,
                        confirmed_overload=confirmed,
                    )
                )
                rows.append(
                    {
                        "video": str(video_path),
                        "frame": frame_index,
                        "timestamp_sec": f"{timestamp_sec:.3f}",
                        "rider_track_id": track_id,
                        "rider_conf": f"{rider.confidence:.3f}",
                        "rider_bbox": " ".join(f"{v:.1f}" for v in rider.bbox),
                        "in_valid_region": int(in_valid_region),
                        "judgment_updated": int(judgment_updated),
                        "occupant_count": person_evidence_count,
                        "person_evidence_count": person_evidence_count,
                        "head_count": head_count,
                        "helmeted_count": helmeted_count,
                        "no_helmet_count": no_helmet_count,
                        "paired_head_helmet_count": paired_head_helmet_count,
                        "unpaired_head_count": unpaired_head_count,
                        "unpaired_helmet_count": unpaired_helmet_count,
                        "matched_occupant_ids": "",
                        "match_scores": "",
                        "raw_overload": int(raw_overload),
                        "raw_overload_candidate": int(raw_overload_candidate),
                        "confirmed_overload": int(confirmed),
                        "helmet_status": "NO_HELMET" if no_helmet_count else "HELMETED" if helmeted_count else "UNKNOWN",
                        "elapsed_ms": f"{elapsed_ms:.3f}",
                        "fps": f"{1000.0 / elapsed_ms:.3f}" if elapsed_ms > 0 else "",
                    }
                )

            frames_with_raw += int(has_raw)
            frames_with_confirmed += int(has_confirmed)
            frames_with_no_helmet += int(has_no_helmet)
            avg_ms_30 = sum(timing_ms[-30:]) / min(len(timing_ms), 30)
            inst_fps = 1000.0 / avg_ms_30 if avg_ms_30 > 0 else 0.0
            annotated = draw_wjh_frame(frame, occupants, frame_results, frame_index, timestamp_sec, f"fps={inst_fps:.1f}")
            annotated = self._draw_roi_overlay(annotated)
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
                        f"processed={processed}/{total_frames} wall_fps={wall_fps:.1f} infer_fps={inst_fps:.1f} "
                        f"riders={total_riders} raw_evidence={total_occupants} raw_frames={frames_with_raw} "
                        f"confirmed_frames={frames_with_confirmed} no_helmet_frames={frames_with_no_helmet} "
                        f"confirmed_tracks={len(overload_track_ids)}"
                    ),
                    "track_results": self._format_track_summary(track_summary),
                }
            )

        cap.release()
        if writer is not None:
            writer.release()
        self._write_frame_csv(frame_csv_path, rows)
        avg_ms = sum(timing_ms) / len(timing_ms) if timing_ms else 0.0
        self._write_summary_csv(summary_csv_path, video_path, processed, total_frames, source_fps, total_riders, total_occupants, frames_with_raw, frames_with_confirmed, frames_with_no_helmet, len(overload_track_ids), avg_ms, str(video_out_path) if writer is not None else "")
        self._put_ui({"type": "status", "text": f"Saved CSV: {summary_csv_path}"})

    def _update_track_summary(self, track_summary, track_id, raw_overload, confirmed_overload, occupant_count, helmeted_count, no_helmet_count, frame_index, overload_status) -> None:
        key = str(track_id)
        state = track_summary.setdefault(
            key,
            {
                "seen_frames": 0,
                "raw_frames": 0,
                "confirmed_frames": 0,
                "ever_confirmed": False,
                "last_status": "NORMAL",
                "last_occupant_count": 0,
                "last_helmeted_count": 0,
                "last_no_helmet_count": 0,
                "no_helmet_frames": 0,
                "last_frame": 0,
            },
        )
        state["seen_frames"] = int(state["seen_frames"]) + 1
        state["raw_frames"] = int(state["raw_frames"]) + int(raw_overload)
        state["confirmed_frames"] = int(state["confirmed_frames"]) + int(confirmed_overload)
        state["ever_confirmed"] = bool(state["ever_confirmed"]) or confirmed_overload
        state["last_status"] = overload_status
        state["last_occupant_count"] = occupant_count
        state["last_helmeted_count"] = helmeted_count
        state["last_no_helmet_count"] = no_helmet_count
        state["no_helmet_frames"] = int(state["no_helmet_frames"]) + int(no_helmet_count > 0)
        state["last_frame"] = frame_index
        state["raw_overload_ratio"] = int(state["raw_frames"]) / max(1, int(state["seen_frames"]))

    def _format_track_summary(self, track_summary) -> List[str]:
        lines = ["ID        result      seen  raw%  conf persons  helm  nohlm  last"]
        for track_id, state in sorted(track_summary.items(), key=lambda item: (not bool(item[1]["ever_confirmed"]), -int(item[1]["confirmed_frames"]), str(item[0]))):
            if str(track_id).startswith("U"):
                result = "UNCERTAIN"
            elif bool(state["ever_confirmed"]):
                result = "CONFIRMED"
            elif str(state.get("last_status", "")) == "SUSPECTED" or int(state["raw_frames"]) > 0:
                result = "SUSPECTED"
            else:
                result = "NORMAL"
            lines.append(
                f"{track_id:<9} {result:<10} {int(state['seen_frames']):>4} "
                f"{float(state.get('raw_overload_ratio', 0.0)) * 100:>4.0f} {int(state['confirmed_frames']):>5} "
                f"{int(state['last_occupant_count']):>6} {int(state['last_helmeted_count']):>5} "
                f"{int(state['last_no_helmet_count']):>6} {int(state['last_frame']):>5}"
            )
        return lines[:101]

    def _write_frame_csv(self, path: Path, rows: List[Dict[str, object]]) -> None:
        fieldnames = [
            "video",
            "frame",
            "timestamp_sec",
            "rider_track_id",
            "rider_conf",
            "rider_bbox",
            "in_valid_region",
            "judgment_updated",
            "occupant_count",
            "person_evidence_count",
            "head_count",
            "helmeted_count",
            "no_helmet_count",
            "paired_head_helmet_count",
            "unpaired_head_count",
            "unpaired_helmet_count",
            "matched_occupant_ids",
            "match_scores",
            "raw_overload",
            "raw_overload_candidate",
            "confirmed_overload",
            "helmet_status",
            "elapsed_ms",
            "fps",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary_csv(self, path: Path, video_path: Path, processed: int, total_frames: int, source_fps: float, total_riders: int, total_occupants: int, frames_with_raw: int, frames_with_confirmed: int, frames_with_no_helmet: int, confirmed_tracks: int, avg_ms: float, saved_video: str) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "video",
                    "model",
                    "processed_frames",
                    "source_frames",
                    "source_fps",
                    "imgsz",
                    "conf",
                    "head_conf",
                    "helmet_conf",
                    "match_thresh",
                    "confirm_frames",
                    "total_rider_detections",
                    "total_occupant_detections",
                    "frames_with_raw_overload",
                    "frames_with_confirmed_overload",
                    "frames_with_no_helmet",
                    "confirmed_overload_tracks",
                    "avg_elapsed_ms",
                    "avg_fps",
                    "saved_video",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "video": str(video_path),
                    "model": self.model_path.get(),
                    "processed_frames": processed,
                    "source_frames": total_frames,
                    "source_fps": f"{source_fps:.3f}",
                    "imgsz": self.imgsz.get(),
                    "conf": self.conf.get(),
                    "head_conf": self.head_conf.get(),
                    "helmet_conf": self.helmet_conf.get(),
                    "match_thresh": self.match_thresh.get(),
                    "confirm_frames": self.confirm_frames.get(),
                    "total_rider_detections": total_riders,
                    "total_occupant_detections": total_occupants,
                    "frames_with_raw_overload": frames_with_raw,
                    "frames_with_confirmed_overload": frames_with_confirmed,
                    "frames_with_no_helmet": frames_with_no_helmet,
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
        self.last_display_frame = frame.copy()
        frame = self._draw_roi_overlay(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        label_w = max(640, self.video_label.winfo_width())
        label_h = max(360, self.video_label.winfo_height())
        img.thumbnail((label_w, label_h))
        self.display_image_size = img.size
        self.display_frame_shape = frame.shape[:2]
        self.display_offset = ((label_w - img.size[0]) // 2, (label_h - img.size[1]) // 2)
        self.current_photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.current_photo)


def main() -> None:
    root = Tk()
    WjhAssociationGuiLjt(root)
    root.mainloop()


if __name__ == "__main__":
    main()
