#!/usr/bin/env python3
"""
Tkinter annotation UI for person-to-vehicle match CSVs and frame-level ID fixes.

Usage:
    python tools/annotate_match_samples.py input.mp4 data/corrections/samples.csv
    python tools/annotate_match_samples.py input.mp4 input-yolov8n-overload-ljt_frames.csv

Keyboard:
    match-sample mode:
    1: mark match_label=1
    0: mark match_label=0
    ID-correction mode:
    Return: apply typed correct_vehicle_id
    Helmet-correction mode:
    h: mark HELMETED
    n: mark NO_HELMET
    u: mark UNKNOWN
    s: skip current row
    Left/Right: previous/next row
    Ctrl+S: save
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    DISABLED,
    END,
    HORIZONTAL,
    LEFT,
    NORMAL,
    RIGHT,
    TOP,
    Button,
    Canvas,
    Entry,
    Frame,
    Label,
    Scale,
    Scrollbar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from typing import Dict, List, Optional


class MatchSampleAnnotator:
    """Small GUI for labeling person-vehicle candidate pairs."""

    def __init__(self, root: Tk, video_path: Path, csv_path: Path, output_path: Optional[Path] = None):
        # Author: You Pinzhen - annotation GUI for correcting match labels and vehicle IDs.
        import cv2

        self.cv2 = cv2
        self.root = root
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_path = output_path or csv_path
        self.rows: List[Dict[str, str]] = []
        self.fieldnames: List[str] = []
        self.mode = "match"
        self.index = 0
        self._loading_row = False
        self.cap = cv2.VideoCapture(str(video_path))
        self.current_photo = None
        self.frame_cache: Dict[int, object] = {}

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._load_csv()
        if not self.rows:
            raise RuntimeError(f"No rows found in CSV: {csv_path}")

        self.status_text = StringVar()
        self.meta_text = StringVar()
        self.label_text = StringVar()
        self.correct_vehicle_id = StringVar()
        self.correct_helmet_status = StringVar()
        self.correct_helmet_statuses = StringVar()
        self.correct_no_helmet_count = StringVar()
        self.error_type = StringVar()
        self.notes = StringVar()
        self.batch_start_frame = StringVar()
        self.batch_end_frame = StringVar()
        self.batch_from_id = StringVar()
        self.batch_to_id = StringVar()
        self.batch_status_text = StringVar(value="Batch ID fix: unset")
        self.region_coords = StringVar()
        self.region_stats_text = StringVar(value="Region stats: unset")
        self._region_bbox: Optional[List[float]] = None
        self._image_offset = (0, 0)
        self._image_scale = 1.0
        self._render_size = (1, 1)
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_rect_id: Optional[int] = None

        self.root.title("Video CSV Annotator")
        self.root.geometry("1180x860")
        self._build_layout()
        self._bind_keys()
        self._bind_canvas_region_selection()
        self._jump_to_first_unlabeled()
        self._render_current()

    def _load_csv(self) -> None:
        with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = [dict(row) for row in reader]

        if {"frame_index", "vehicle_bbox", "person_bbox", "match_label"}.issubset(self.fieldnames):
            self.mode = "match"
        elif {"frame", "vehicle_bbox", "vehicle_track_id"}.issubset(self.fieldnames):
            self.mode = "id"
        elif {"frame", "rider_bbox", "rider_track_id", "helmet_status"}.issubset(self.fieldnames):
            self.mode = "helmet"
        elif {"frame", "target_id", "helmet_status"}.issubset(self.fieldnames):
            self.mode = "helmet_target"
        elif {"frame", "helmet_statuses"}.issubset(self.fieldnames):
            self.mode = "helmet_frame"
        else:
            raise RuntimeError(
                "Unsupported CSV schema. Expected match sample columns "
                "(frame_index, vehicle_bbox, person_bbox, match_label) or frame result columns "
                "(frame, vehicle_bbox, vehicle_track_id) or helmet frame result columns "
                "(frame, rider_bbox, rider_track_id, helmet_status) or cross-validation target columns "
                "(frame, target_id, helmet_status) or cross-validation frame columns "
                "(frame, helmet_statuses)."
            )

        for optional in (
            "correct_vehicle_id",
            "final_vehicle_id",
            "id_fix_status",
            "correct_helmet_status",
            "final_helmet_status",
            "correct_helmet_statuses",
            "final_helmet_statuses",
            "correct_no_helmet_count",
            "final_no_helmet_count",
            "helmet_fix_status",
            "error_type",
            "notes",
        ):
            if optional not in self.fieldnames:
                self.fieldnames.append(optional)
                for row in self.rows:
                    row[optional] = ""

    def _build_layout(self) -> None:
        top = Frame(self.root, padx=8, pady=8)
        top.pack(side=TOP, fill=BOTH, expand=True)

        self.canvas = Canvas(top, width=920, height=690, bg="#111111", highlightthickness=0)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        side_shell = Frame(top, padx=10)
        side_shell.pack(side=RIGHT, fill="y")
        side_canvas = Canvas(side_shell, width=270, highlightthickness=0)
        side_scrollbar = Scrollbar(side_shell, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side_scrollbar.pack(side=RIGHT, fill="y")
        side_canvas.pack(side=LEFT, fill="y", expand=False)
        side = Frame(side_canvas)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind(
            "<Configure>",
            lambda _event: side_canvas.configure(scrollregion=side_canvas.bbox("all")),
        )
        side_canvas.bind(
            "<Configure>",
            lambda event: side_canvas.itemconfigure(side_window, width=event.width),
        )
        self._bind_mousewheel(side_canvas)

        Label(side, textvariable=self.status_text, justify=LEFT, wraplength=230).pack(anchor="w", pady=6)
        Label(side, textvariable=self.meta_text, justify=LEFT, wraplength=230).pack(anchor="w", pady=6)
        Label(side, textvariable=self.label_text, justify=LEFT, wraplength=230).pack(anchor="w", pady=6)

        Button(side, text="属于 (1)", command=lambda: self.mark("1"), width=22).pack(pady=4)
        Button(side, text="不属于 (0)", command=lambda: self.mark("0"), width=22).pack(pady=4)
        Button(side, text="有头盔 (H)", command=lambda: self.mark_helmet("HELMETED"), width=22).pack(pady=4)
        Button(side, text="无头盔 (N)", command=lambda: self.mark_helmet("NO_HELMET"), width=22).pack(pady=4)
        Button(side, text="头盔未知 (U)", command=lambda: self.mark_helmet("UNKNOWN"), width=22).pack(pady=4)
        Button(side, text="应用 ID (Enter)", command=self.apply_id_fix, width=22).pack(pady=4)
        Button(side, text="沿用上一 ID", command=self.copy_previous_id, width=22).pack(pady=4)
        Button(side, text="使用原 ID", command=self.use_original_id, width=22).pack(pady=4)
        Button(side, text="清空修正", command=self.clear_id_fix, width=22).pack(pady=4)
        Button(side, text="跳过 (S)", command=self.skip, width=22).pack(pady=4)
        Button(side, text="上一条", command=self.previous_row, width=22).pack(pady=4)
        Button(side, text="下一条", command=self.next_row, width=22).pack(pady=4)
        Button(side, text="下一个未标注", command=self.next_unlabeled, width=22).pack(pady=4)

        Label(side, text="correct_vehicle_id").pack(anchor="w", pady=(14, 2))
        self.correct_vehicle_entry = Entry(side, textvariable=self.correct_vehicle_id, width=26)
        self.correct_vehicle_entry.pack(anchor="w")

        Label(side, text="correct_helmet_status").pack(anchor="w", pady=(10, 2))
        self.correct_helmet_status_entry = Entry(side, textvariable=self.correct_helmet_status, width=26)
        self.correct_helmet_status_entry.pack(anchor="w")

        Label(side, text="correct_helmet_statuses").pack(anchor="w", pady=(10, 2))
        self.correct_helmet_statuses_entry = Entry(side, textvariable=self.correct_helmet_statuses, width=26)
        self.correct_helmet_statuses_entry.pack(anchor="w")

        Label(side, text="correct_no_helmet_count").pack(anchor="w", pady=(10, 2))
        self.correct_no_helmet_count_entry = Entry(side, textvariable=self.correct_no_helmet_count, width=26)
        self.correct_no_helmet_count_entry.pack(anchor="w")

        Label(side, text="error_type").pack(anchor="w", pady=(10, 2))
        self.error_type_entry = Entry(side, textvariable=self.error_type, width=26)
        self.error_type_entry.pack(anchor="w")

        Label(side, text="notes").pack(anchor="w", pady=(10, 2))
        self.notes_entry = Entry(side, textvariable=self.notes, width=26)
        self.notes_entry.pack(anchor="w")

        Label(side, text="batch ID fix").pack(anchor="w", pady=(16, 2))
        Label(side, text="start_frame").pack(anchor="w")
        Entry(side, textvariable=self.batch_start_frame, width=26).pack(anchor="w")
        Label(side, text="end_frame").pack(anchor="w", pady=(6, 0))
        Entry(side, textvariable=self.batch_end_frame, width=26).pack(anchor="w")
        Label(side, text="from_id").pack(anchor="w", pady=(6, 0))
        Entry(side, textvariable=self.batch_from_id, width=26).pack(anchor="w")
        Label(side, text="to_id").pack(anchor="w", pady=(6, 0))
        Entry(side, textvariable=self.batch_to_id, width=26).pack(anchor="w")
        Button(side, text="填当前帧/ID", command=self.prefill_batch_id_fix, width=22).pack(pady=4)
        Button(side, text="批量替换 ID", command=self.apply_batch_id_fix, width=22).pack(pady=4)
        Label(side, textvariable=self.batch_status_text, justify=LEFT, wraplength=230).pack(anchor="w", pady=4)

        Button(side, text="保存 (Ctrl+S)", command=self.save, width=22).pack(pady=(16, 4))
        Button(side, text="另存为", command=self.save_as, width=22).pack(pady=4)

        Label(side, text="region x1,y1,x2,y2").pack(anchor="w", pady=(16, 2))
        self.region_entry = Entry(side, textvariable=self.region_coords, width=26)
        self.region_entry.pack(anchor="w")
        Button(side, text="统计区域车辆", command=self.compute_region_stats, width=22).pack(pady=4)
        Button(side, text="清空区域", command=self.clear_region, width=22).pack(pady=4)
        Button(side, text="导出区域统计", command=self.export_region_stats, width=22).pack(pady=4)
        Label(side, textvariable=self.region_stats_text, justify=LEFT, wraplength=230).pack(anchor="w", pady=6)

        bottom = Frame(self.root, padx=8, pady=6)
        bottom.pack(side=BOTTOM, fill="x")
        self.slider = Scale(
            bottom,
            from_=0,
            to=max(0, len(self.rows) - 1),
            orient=HORIZONTAL,
            command=self._on_slider,
            showvalue=True,
        )
        self.slider.pack(fill="x")

    def _bind_canvas_region_selection(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_region_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_region_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_region_drag_end)

    def _bind_mousewheel(self, widget: Canvas) -> None:
        def on_mousewheel(event) -> None:
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")

        widget.bind("<Enter>", lambda _event: widget.bind_all("<MouseWheel>", on_mousewheel))
        widget.bind("<Leave>", lambda _event: widget.unbind_all("<MouseWheel>"))

    def _bind_keys(self) -> None:
        self.root.bind("1", lambda _event: self.mark("1"))
        self.root.bind("0", lambda _event: self.mark("0"))
        self.root.bind("h", lambda _event: self.mark_helmet("HELMETED"))
        self.root.bind("n", lambda _event: self.mark_helmet("NO_HELMET"))
        self.root.bind("u", lambda _event: self.mark_helmet("UNKNOWN"))
        self.root.bind("<Return>", lambda _event: self.apply_id_fix())
        self.root.bind("s", lambda _event: self.skip())
        self.root.bind("<Right>", lambda _event: self.next_row())
        self.root.bind("<Left>", lambda _event: self.previous_row())
        self.root.bind("<Control-s>", lambda _event: self.save())

    def _jump_to_first_unlabeled(self) -> None:
        for idx, row in enumerate(self.rows):
            if self.mode == "match" and not row.get("match_label", "").strip():
                self.index = idx
                return
            if self.mode == "id" and not row.get("correct_vehicle_id", "").strip():
                self.index = idx
                return
            if self.mode in {"helmet", "helmet_target"} and not row.get("correct_helmet_status", "").strip():
                self.index = idx
                return
            if self.mode == "helmet_frame" and not row.get("correct_helmet_statuses", "").strip():
                self.index = idx
                return

    def _render_current(self) -> None:
        from PIL import Image, ImageDraw, ImageFont, ImageTk

        row = self.rows[self.index]
        frame_index = self._row_frame_index(row)
        frame = self._read_frame(frame_index)
        if frame is None:
            messagebox.showerror("Error", f"Cannot read frame {frame_index}")
            return

        main_bbox = None
        if self.mode == "helmet" and row.get("rider_bbox", "").strip():
            main_bbox = _parse_bbox(row["rider_bbox"])
        elif self.mode not in {"helmet", "helmet_target", "helmet_frame"}:
            main_bbox = _parse_bbox(row["vehicle_bbox"])
        person_bbox = _parse_bbox(row["person_bbox"]) if self.mode == "match" else None
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        canvas_w = max(1, self.canvas.winfo_width() or 920)
        canvas_h = max(1, self.canvas.winfo_height() or 690)
        scale = min(canvas_w / image.width, canvas_h / image.height)
        render_w = max(1, int(image.width * scale))
        render_h = max(1, int(image.height * scale))
        offset_x = (canvas_w - render_w) // 2
        offset_y = (canvas_h - render_h) // 2
        self._image_offset = (offset_x, offset_y)
        self._image_scale = scale
        self._render_size = (render_w, render_h)
        image = image.resize((render_w, render_h))

        draw = ImageDraw.Draw(image)
        if self._region_bbox is not None:
            _draw_region(draw, self._region_bbox, scale)
        if self.mode in {"helmet", "helmet_target"}:
            original_status = row.get("helmet_status", "")
            final_status = _row_helmet_status(row)
            target_label = row.get("rider_track_id", "") or row.get("target_id", "")
            rider_label = f"target {target_label} {original_status}"
            if final_status and final_status != original_status:
                rider_label += f" -> {final_status}"
            if main_bbox is not None:
                color = "lime" if final_status == "HELMETED" else "orange" if final_status == "NO_HELMET" else "yellow"
                _draw_bbox(draw, main_bbox, scale, color, rider_label)
        elif self.mode == "helmet_frame":
            pass
        else:
            vehicle_label = f"vehicle ID {row.get('vehicle_track_id', '')}"
            corrected = row.get("correct_vehicle_id", "").strip()
            if corrected:
                vehicle_label += f" -> {corrected}"
            _draw_bbox(draw, main_bbox, scale, "lime", vehicle_label)
        if person_bbox is not None:
            _draw_bbox(draw, person_bbox, scale, "cyan", "person")
            _draw_pair_line(draw, main_bbox, person_bbox, scale)

        try:
            font = ImageFont.truetype("Arial.ttf", 15)
        except OSError:
            font = ImageFont.load_default()
        if self.mode == "match":
            label = f"match_label={row.get('match_label', '') or 'unlabeled'}"
        elif self.mode in {"helmet", "helmet_target"}:
            label = f"helmet {row.get('helmet_status', '')} -> {row.get('correct_helmet_status', '') or '(unset)'}"
        elif self.mode == "helmet_frame":
            label = f"helmet_statuses -> {row.get('correct_helmet_statuses', '') or '(unset)'}"
        else:
            label = f"ID {row.get('vehicle_track_id', '')} -> {row.get('correct_vehicle_id', '') or '(unset)'}"
        draw.rectangle((8, 8, 260, 34), fill=(0, 0, 0))
        draw.text((14, 13), label, fill="white", font=font)

        self.current_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.current_photo)

        self._loading_row = True
        try:
            self.correct_vehicle_id.set(row.get("correct_vehicle_id", ""))
            self.correct_helmet_status.set(row.get("correct_helmet_status", ""))
            self.correct_helmet_statuses.set(row.get("correct_helmet_statuses", ""))
            self.correct_no_helmet_count.set(row.get("correct_no_helmet_count", ""))
            self.error_type.set(row.get("error_type", ""))
            self.notes.set(row.get("notes", ""))
        finally:
            self._loading_row = False
        self.slider.set(self.index)
        self._update_text()

    def _on_region_drag_start(self, event) -> None:
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None:
            return
        self._drag_start = (event.x, event.y)
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="red",
            width=3,
        )

    def _on_region_drag_move(self, event) -> None:
        if self._drag_start is None or self._drag_rect_id is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = self._clamp_canvas_to_image(event.x, event.y)
        self.canvas.coords(self._drag_rect_id, x0, y0, x1, y1)

    def _on_region_drag_end(self, event) -> None:
        if self._drag_start is None:
            return
        start = self._canvas_to_image_point(*self._drag_start)
        end = self._canvas_to_image_point(*self._clamp_canvas_to_image(event.x, event.y))
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        if start is None or end is None:
            return
        x1, y1 = start
        x2, y2 = end
        region = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if region[2] - region[0] < 5 or region[3] - region[1] < 5:
            return
        self.region_coords.set(",".join(f"{value:.1f}" for value in region))
        self.compute_region_stats()

    def _canvas_to_image_point(self, x: int, y: int) -> Optional[tuple[float, float]]:
        offset_x, offset_y = self._image_offset
        render_w, render_h = self._render_size
        if not (offset_x <= x <= offset_x + render_w and offset_y <= y <= offset_y + render_h):
            return None
        return ((x - offset_x) / self._image_scale, (y - offset_y) / self._image_scale)

    def _clamp_canvas_to_image(self, x: int, y: int) -> tuple[int, int]:
        offset_x, offset_y = self._image_offset
        render_w, render_h = self._render_size
        return (
            max(offset_x, min(offset_x + render_w, x)),
            max(offset_y, min(offset_y + render_h, y)),
        )

    def _row_frame_index(self, row: Dict[str, str]) -> int:
        key = "frame_index" if self.mode == "match" else "frame"
        return int(float(row[key]))

    def _read_frame(self, frame_index: int):
        if frame_index in self.frame_cache:
            return self.frame_cache[frame_index].copy()

        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = self.cap.read()
        if not ok:
            return None
        if len(self.frame_cache) > 30:
            self.frame_cache.clear()
        self.frame_cache[frame_index] = frame.copy()
        return frame

    def _update_text(self) -> None:
        row = self.rows[self.index]
        if self.mode == "match":
            labeled = sum(1 for item in self.rows if item.get("match_label", "").strip() in {"0", "1"})
        elif self.mode in {"helmet", "helmet_target"}:
            labeled = sum(1 for item in self.rows if item.get("correct_helmet_status", "").strip())
        elif self.mode == "helmet_frame":
            labeled = sum(1 for item in self.rows if item.get("correct_helmet_statuses", "").strip())
        else:
            labeled = sum(1 for item in self.rows if item.get("correct_vehicle_id", "").strip())
        self.status_text.set(
            f"CSV: {self.csv_path.name}\n"
            f"Video: {self.video_path.name}\n"
            f"Mode: {self.mode}\n"
            f"Row: {self.index + 1}/{len(self.rows)}\n"
            f"Edited: {labeled}/{len(self.rows)}"
        )
        self.meta_text.set(
            f"Frame: {self._row_frame_index(row)}\n"
            f"Time: {row.get('timestamp', row.get('timestamp_sec', ''))}s\n"
            f"vehicle_track_id: {row.get('vehicle_track_id', '')}\n"
            f"rider_track_id: {row.get('rider_track_id', '')}\n"
            f"target_id: {row.get('target_id', '')}\n"
            f"wjh_id: {row.get('wjh_id', '')}\n"
            f"yolo_id: {row.get('yolo_id', '')}\n"
            f"person_track_id: {row.get('person_track_id', '')}\n"
            f"heuristic_match: {row.get('heuristic_match', '')}\n"
            f"raw_overload: {row.get('raw_overload', '')}\n"
            f"confirmed_overload: {row.get('confirmed_overload', '')}"
        )
        self.label_text.set(
            f"match_label: {row.get('match_label', '') or '(empty)'}\n"
            f"correct_vehicle_id: {row.get('correct_vehicle_id', '') or '(empty)'}\n"
            f"final_vehicle_id: {row.get('final_vehicle_id', '') or '(empty)'}\n"
            f"helmet_status: {row.get('helmet_status', '') or '(empty)'}\n"
            f"correct_helmet_status: {row.get('correct_helmet_status', '') or '(empty)'}\n"
            f"final_helmet_status: {row.get('final_helmet_status', '') or '(empty)'}\n"
            f"helmet_statuses: {row.get('helmet_statuses', '') or '(empty)'}\n"
            f"correct_helmet_statuses: {row.get('correct_helmet_statuses', '') or '(empty)'}\n"
            f"final_helmet_statuses: {row.get('final_helmet_statuses', '') or '(empty)'}\n"
            f"no_helmet_count: {row.get('no_helmet_count', '') or '(empty)'}\n"
            f"final_no_helmet_count: {row.get('final_no_helmet_count', '') or '(empty)'}\n"
            f"vehicle_conf: {row.get('vehicle_conf', '')}\n"
            f"person_conf: {row.get('person_conf', '')}"
        )

    def mark(self, label: str) -> None:
        if self.mode != "match":
            return
        self._sync_edit_fields_to_row()
        self.rows[self.index]["match_label"] = label
        self.save()
        self.next_unlabeled(fallback_next=True)

    def mark_helmet(self, status: str) -> None:
        if self.mode not in {"helmet", "helmet_target", "helmet_frame"}:
            return
        self._sync_edit_fields_to_row()
        row = self.rows[self.index]
        if self.mode == "helmet_frame":
            _set_row_helmet_statuses_fix(row, status)
        else:
            _set_row_helmet_fix(row, status)
        self._load_current_row_edit_fields()
        self.save()
        self.next_unlabeled(fallback_next=True)

    def apply_id_fix(self) -> None:
        self._sync_edit_fields_to_row()
        row = self.rows[self.index]
        corrected = row.get("correct_vehicle_id", "").strip()
        _set_row_correct_vehicle_id(row, corrected)
        self.save()
        self.next_row()

    def copy_previous_id(self) -> None:
        previous_id = ""
        for idx in range(self.index - 1, -1, -1):
            previous_id = self.rows[idx].get("correct_vehicle_id", "").strip() or self.rows[idx].get("final_vehicle_id", "").strip()
            if previous_id:
                break
        if previous_id:
            self.correct_vehicle_id.set(previous_id)
            self.apply_id_fix()

    def use_original_id(self) -> None:
        row = self.rows[self.index]
        self.correct_vehicle_id.set(row.get("vehicle_track_id", ""))
        self.apply_id_fix()

    def clear_id_fix(self) -> None:
        row = self.rows[self.index]
        self.correct_vehicle_id.set("")
        row["correct_vehicle_id"] = ""
        row["final_vehicle_id"] = ""
        row["id_fix_status"] = ""
        self.save()
        self._render_current()

    def prefill_batch_id_fix(self) -> None:
        if self.mode != "id":
            messagebox.showinfo("ID mode only", "Batch ID fix is only available for frame result CSVs.")
            return
        self._sync_edit_fields_to_row()
        row = self.rows[self.index]
        frame = str(self._row_frame_index(row))
        current_id = row.get("correct_vehicle_id", "").strip() or row.get("final_vehicle_id", "").strip()
        if not current_id:
            current_id = row.get("vehicle_track_id", "").strip()
        self.batch_start_frame.set(self.batch_start_frame.get().strip() or frame)
        self.batch_end_frame.set(self.batch_end_frame.get().strip() or frame)
        self.batch_from_id.set(current_id)
        self.batch_to_id.set(self.batch_to_id.get().strip() or current_id)
        self.batch_status_text.set(f"Batch ID fix: prefilled frame {frame}, ID {current_id or '(empty)'}")

    def apply_batch_id_fix(self) -> None:
        # Author: You Pinzhen - batch ID correction to keep vehicle identities aligned across frame ranges.
        if self.mode != "id":
            messagebox.showinfo("ID mode only", "Batch ID fix is only available for frame result CSVs.")
            return
        self._sync_edit_fields_to_row()

        try:
            start_frame = int(float(self.batch_start_frame.get().strip()))
            end_frame = int(float(self.batch_end_frame.get().strip()))
        except ValueError:
            messagebox.showerror("Invalid frame range", "start_frame and end_frame must be numbers.")
            return
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        from_id = self.batch_from_id.get().strip()
        to_id = self.batch_to_id.get().strip()
        if not from_id or not to_id:
            messagebox.showerror("Invalid ID", "from_id and to_id cannot be empty.")
            return

        changed = 0
        touched_frames: set[int] = set()
        for row in self.rows:
            frame = self._row_frame_index(row)
            if frame < start_frame or frame > end_frame:
                continue
            if not _row_matches_vehicle_id(row, from_id):
                continue
            _set_row_correct_vehicle_id(row, to_id, status="batch_corrected")
            changed += 1
            touched_frames.add(frame)

        self.batch_status_text.set(
            f"Batch ID fix: {changed} rows, {len(touched_frames)} frames\n"
            f"{from_id} -> {to_id}, frames {start_frame}-{end_frame}"
        )
        if changed == 0:
            messagebox.showwarning(
                "No rows changed",
                f"No rows matched ID {from_id} between frames {start_frame} and {end_frame}.",
            )
            return

        self._load_current_row_edit_fields()
        self.save()
        self._render_current()
        messagebox.showinfo(
            "Batch ID fix applied",
            f"Changed {changed} rows on {len(touched_frames)} frames.\n"
            f"ID {from_id} -> {to_id}, frames {start_frame}-{end_frame}.",
        )

    def skip(self) -> None:
        self._sync_edit_fields_to_row()
        self.next_row()

    def previous_row(self) -> None:
        self._sync_edit_fields_to_row()
        self.index = max(0, self.index - 1)
        self._render_current()

    def next_row(self) -> None:
        self._sync_edit_fields_to_row()
        self.index = min(len(self.rows) - 1, self.index + 1)
        self._render_current()

    def next_unlabeled(self, fallback_next: bool = False) -> None:
        self._sync_edit_fields_to_row()
        for idx in range(self.index + 1, len(self.rows)):
            if self.mode == "match" and not self.rows[idx].get("match_label", "").strip():
                self.index = idx
                self._render_current()
                return
            if self.mode == "id" and not self.rows[idx].get("correct_vehicle_id", "").strip():
                self.index = idx
                self._render_current()
                return
            if self.mode in {"helmet", "helmet_target"} and not self.rows[idx].get("correct_helmet_status", "").strip():
                self.index = idx
                self._render_current()
                return
            if self.mode == "helmet_frame" and not self.rows[idx].get("correct_helmet_statuses", "").strip():
                self.index = idx
                self._render_current()
                return
        if fallback_next:
            self.index = min(len(self.rows) - 1, self.index + 1)
        self._render_current()

    def save(self) -> None:
        self._sync_edit_fields_to_row()
        with self.output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        self._update_text()

    def save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save annotated CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.output_path = Path(path)
        self.save()
        messagebox.showinfo("Saved", f"Saved to {self.output_path}")

    def compute_region_stats(self) -> None:
        # Author: You Pinzhen - manual region counting support for vehicle and overload statistics.
        try:
            region = _parse_region(self.region_coords.get())
        except ValueError as exc:
            messagebox.showerror("Invalid region", str(exc))
            return
        self._region_bbox = region
        stats = self._region_stats(region)
        self.region_stats_text.set(_format_region_stats(stats))
        self._render_current()

    def clear_region(self) -> None:
        self._region_bbox = None
        self.region_coords.set("")
        self.region_stats_text.set("Region stats: unset")
        self._render_current()

    def export_region_stats(self) -> None:
        if self._region_bbox is None:
            try:
                self._region_bbox = _parse_region(self.region_coords.get())
            except ValueError as exc:
                messagebox.showerror("Invalid region", str(exc))
                return
        stats = self._region_stats(self._region_bbox)
        default_path = self.csv_path.with_name(f"{self.csv_path.stem}_region_stats.csv")
        path = filedialog.asksaveasfilename(
            title="Save region stats",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "video",
                "source_csv",
                "region_bbox",
                "vehicle_count",
                "overload_vehicle_count",
                "row_hits",
                "frame_hits",
                "vehicle_ids",
                "overload_vehicle_ids",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "video": str(self.video_path),
                    "source_csv": str(self.csv_path),
                    "region_bbox": " ".join(f"{value:.1f}" for value in self._region_bbox),
                    "vehicle_count": stats["vehicle_count"],
                    "overload_vehicle_count": stats["overload_vehicle_count"],
                    "row_hits": stats["row_hits"],
                    "frame_hits": stats["frame_hits"],
                    "vehicle_ids": " ".join(stats["vehicle_ids"]),
                    "overload_vehicle_ids": " ".join(stats["overload_vehicle_ids"]),
                }
            )
        messagebox.showinfo("Saved", f"Saved to {path}")

    def _region_stats(self, region: List[float]) -> Dict[str, object]:
        vehicle_ids: set[str] = set()
        overload_vehicle_ids: set[str] = set()
        frame_hits: set[int] = set()
        row_hits = 0

        for row in self.rows:
            if "vehicle_bbox" not in row:
                continue
            try:
                vehicle_bbox = _parse_bbox(row["vehicle_bbox"])
            except Exception:
                continue
            if not _box_touches_region(vehicle_bbox, region):
                continue
            row_hits += 1
            frame_hits.add(self._row_frame_index(row))
            vehicle_id = _row_vehicle_identity(row)
            vehicle_ids.add(vehicle_id)
            if _row_is_overload(row):
                overload_vehicle_ids.add(vehicle_id)

        return {
            "vehicle_count": len(vehicle_ids),
            "overload_vehicle_count": len(overload_vehicle_ids),
            "row_hits": row_hits,
            "frame_hits": len(frame_hits),
            "vehicle_ids": sorted(vehicle_ids),
            "overload_vehicle_ids": sorted(overload_vehicle_ids),
        }

    def _on_slider(self, value: str) -> None:
        new_index = int(float(value))
        if new_index == self.index:
            return
        self._sync_edit_fields_to_row()
        self.index = max(0, min(len(self.rows) - 1, new_index))
        self._render_current()

    def _sync_edit_fields_to_row(self) -> None:
        if not self.rows or self._loading_row:
            return
        row = self.rows[self.index]
        _set_row_correct_vehicle_id(row, self.correct_vehicle_id.get())
        if self.mode in {"helmet", "helmet_target"}:
            _set_row_helmet_fix(row, self.correct_helmet_status.get(), self.correct_no_helmet_count.get())
        elif self.mode == "helmet_frame":
            _set_row_helmet_statuses_fix(row, self.correct_helmet_statuses.get())
        row["error_type"] = self.error_type.get()
        row["notes"] = self.notes.get()

    def _load_current_row_edit_fields(self) -> None:
        row = self.rows[self.index]
        self._loading_row = True
        try:
            self.correct_vehicle_id.set(row.get("correct_vehicle_id", ""))
            self.correct_helmet_status.set(row.get("correct_helmet_status", ""))
            self.correct_helmet_statuses.set(row.get("correct_helmet_statuses", ""))
            self.correct_no_helmet_count.set(row.get("correct_no_helmet_count", ""))
            self.error_type.set(row.get("error_type", ""))
            self.notes.set(row.get("notes", ""))
        finally:
            self._loading_row = False

    def close(self) -> None:
        self.save()
        self.cap.release()
        self.root.destroy()


def _parse_bbox(raw: str) -> List[float]:
    raw = raw.strip()
    if raw.startswith("["):
        value = json.loads(raw)
        return [float(item) for item in value]
    return [float(item) for item in raw.replace(",", " ").split()]


def _draw_bbox(draw, bbox: List[float], scale: float, color: str, label: str) -> None:
    x1, y1, x2, y2 = [int(value * scale) for value in bbox]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    label_w = max(86, min(240, len(label) * 8 + 12))
    draw.rectangle((x1, max(0, y1 - 22), x1 + label_w, y1), fill=color)
    draw.text((x1 + 4, max(0, y1 - 19)), label, fill="black")


def _draw_region(draw, bbox: List[float], scale: float) -> None:
    x1, y1, x2, y2 = [int(value * scale) for value in bbox]
    draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
    draw.rectangle((x1, max(0, y1 - 24), x1 + 90, y1), fill="red")
    draw.text((x1 + 5, max(0, y1 - 20)), "region", fill="white")


def _draw_pair_line(draw, vehicle_bbox: List[float], person_bbox: List[float], scale: float) -> None:
    vx = int(((vehicle_bbox[0] + vehicle_bbox[2]) / 2) * scale)
    vy = int(((vehicle_bbox[1] + vehicle_bbox[3]) / 2) * scale)
    px = int(((person_bbox[0] + person_bbox[2]) / 2) * scale)
    py = int(((person_bbox[1] + person_bbox[3]) / 2) * scale)
    draw.line((vx, vy, px, py), fill="yellow", width=2)


def _parse_region(raw: str) -> List[float]:
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError("Region must contain four numbers: x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Region must satisfy x2 > x1 and y2 > y1")
    return [x1, y1, x2, y2]


def _box_touches_region(box: List[float], region: List[float]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
        return True
    return _intersection_area(box, region) > 0


def _intersection_area(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _row_vehicle_identity(row: Dict[str, str]) -> str:
    for key in ("final_vehicle_id", "correct_vehicle_id", "vehicle_track_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "unknown"


def _row_matches_vehicle_id(row: Dict[str, str], target_id: str) -> bool:
    target_id = str(target_id).strip()
    return any(str(row.get(key, "")).strip() == target_id for key in ("vehicle_track_id", "correct_vehicle_id", "final_vehicle_id"))


def _set_row_correct_vehicle_id(row: Dict[str, str], corrected_id: str, status: str = "corrected") -> None:
    corrected_id = str(corrected_id).strip()
    original_id = str(row.get("vehicle_track_id", "")).strip()
    row["correct_vehicle_id"] = corrected_id
    row["final_vehicle_id"] = corrected_id or original_id
    if not corrected_id:
        row["id_fix_status"] = ""
    elif corrected_id == original_id:
        row["id_fix_status"] = "original"
    else:
        row["id_fix_status"] = status


def _normalize_helmet_status(status: str) -> str:
    value = str(status).strip().upper()
    aliases = {
        "1": "HELMETED",
        "YES": "HELMETED",
        "Y": "HELMETED",
        "TRUE": "HELMETED",
        "HELMET": "HELMETED",
        "HAS_HELMET": "HELMETED",
        "0": "NO_HELMET",
        "NO": "NO_HELMET",
        "N": "NO_HELMET",
        "FALSE": "NO_HELMET",
        "NOHELMET": "NO_HELMET",
        "NO_HELMET": "NO_HELMET",
        "UNKNOWN": "UNKNOWN",
        "U": "UNKNOWN",
        "": "",
    }
    return aliases.get(value, value)


def _row_helmet_status(row: Dict[str, str]) -> str:
    for key in ("final_helmet_status", "correct_helmet_status", "helmet_status"):
        value = _normalize_helmet_status(row.get(key, ""))
        if value:
            return value
    return "UNKNOWN"


def _default_no_helmet_count_for_status(row: Dict[str, str], status: str) -> str:
    status = _normalize_helmet_status(status)
    if status == "NO_HELMET":
        original_count = str(row.get("no_helmet_count", "")).strip()
        if original_count and original_count not in {"0", "0.0"}:
            return original_count
        return "1"
    if status in {"HELMETED", "UNKNOWN"}:
        return "0"
    return str(row.get("no_helmet_count", "")).strip()


def _set_row_helmet_fix(row: Dict[str, str], corrected_status: str, corrected_no_helmet_count: str = "") -> None:
    corrected_status = _normalize_helmet_status(corrected_status)
    original_status = _normalize_helmet_status(row.get("helmet_status", ""))
    corrected_no_helmet_count = str(corrected_no_helmet_count).strip()
    if corrected_status and not corrected_no_helmet_count:
        corrected_no_helmet_count = _default_no_helmet_count_for_status(row, corrected_status)

    row["correct_helmet_status"] = corrected_status
    row["final_helmet_status"] = corrected_status or original_status
    row["correct_no_helmet_count"] = corrected_no_helmet_count
    row["final_no_helmet_count"] = corrected_no_helmet_count or str(row.get("no_helmet_count", "")).strip()

    if not corrected_status:
        row["helmet_fix_status"] = ""
    elif corrected_status == original_status and str(row.get("correct_no_helmet_count", "")).strip() in {"", str(row.get("no_helmet_count", "")).strip()}:
        row["helmet_fix_status"] = "original"
    else:
        row["helmet_fix_status"] = "corrected"


def _set_row_helmet_statuses_fix(row: Dict[str, str], corrected_statuses: str) -> None:
    corrected_statuses = str(corrected_statuses).strip()
    original_statuses = str(row.get("helmet_statuses", "")).strip()

    if _normalize_helmet_status(corrected_statuses) in {"HELMETED", "NO_HELMET", "UNKNOWN"}:
        corrected_statuses = _replace_statuses_with_single_value(original_statuses, corrected_statuses)

    row["correct_helmet_statuses"] = corrected_statuses
    row["final_helmet_statuses"] = corrected_statuses or original_statuses
    if not corrected_statuses:
        row["helmet_fix_status"] = ""
    elif corrected_statuses == original_statuses:
        row["helmet_fix_status"] = "original"
    else:
        row["helmet_fix_status"] = "corrected"


def _replace_statuses_with_single_value(original_statuses: str, status: str) -> str:
    status = _normalize_helmet_status(status)
    parts = [part for part in str(original_statuses).split() if part.strip()]
    if not parts:
        return status
    replaced = []
    for part in parts:
        if ":" in part:
            target_id, _old_status = part.rsplit(":", 1)
            replaced.append(f"{target_id}:{status}")
        else:
            replaced.append(status)
    return " ".join(replaced)


def _row_is_overload(row: Dict[str, str]) -> bool:
    for key in ("confirmed_overload", "raw_overload"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "true", "yes"}:
            return True
    return False


def _format_region_stats(stats: Dict[str, object]) -> str:
    vehicle_ids = stats["vehicle_ids"]
    overload_ids = stats["overload_vehicle_ids"]
    return (
        "Region stats:\n"
        f"vehicles: {stats['vehicle_count']}\n"
        f"overload vehicles: {stats['overload_vehicle_count']}\n"
        f"rows in region: {stats['row_hits']}\n"
        f"frames in region: {stats['frame_hits']}\n"
        f"vehicle IDs: {_short_list(vehicle_ids)}\n"
        f"overload IDs: {_short_list(overload_ids)}"
    )


def _short_list(values: List[str], limit: int = 10) -> str:
    if not values:
        return "-"
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", ...(+{len(values) - limit})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate match samples or correct frame-level vehicle IDs")
    parser.add_argument("video", help="Input video path used to produce the CSV")
    parser.add_argument("csv", help="match_samples.csv or *yolov8n-overload-ljt_frames.csv")
    parser.add_argument("--output", "-o", help="Optional output CSV; defaults to overwriting input CSV")
    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    output_path = Path(args.output) if args.output else None

    if not video_path.exists():
        print(f"Video does not exist: {video_path}")
        return 1
    if not csv_path.exists():
        print(f"CSV does not exist: {csv_path}")
        return 1

    root = Tk()
    app = MatchSampleAnnotator(root, video_path, csv_path, output_path=output_path)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
