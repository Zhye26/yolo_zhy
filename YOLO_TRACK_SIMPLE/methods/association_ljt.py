from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.pipeline.pipeline_ljt import (
    Detection,
    TrackedDetection,
    iou,
    overload_status_from_stats,
    person_vehicle_match_score,
    update_simple_overload_stats,
)


class AssociationMotTracker:
    def __init__(
        self,
        person_tracker,
        vehicle_tracker,
        confirm_frames: int,
        max_missed: int,
        association_min_hits: int,
        association_lock_frames: int,
        association_unbind_frames: int,
        association_switch_margin: float,
        match_score_fn: Callable[[Detection, Detection], float] = person_vehicle_match_score,
    ):
        self.person_tracker = person_tracker
        self.vehicle_tracker = vehicle_tracker
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
        self.association_min_hits = association_min_hits
        self.association_lock_frames = association_lock_frames
        self.association_unbind_frames = association_unbind_frames
        self.association_switch_margin = association_switch_margin
        self.frame_index = 0
        self.confirmed_track_ids: set[str] = set()
        self.raw_to_stable: Dict[str, Dict[int, str]] = {"person": {}, "vehicle": {}}
        self.stable_objects: Dict[str, Dict[str, object]] = {}
        self.next_person_id = 1
        self.next_vehicle_id = 1
        self.bindings: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.vehicle_states: Dict[str, Dict[str, object]] = {}
        self.archived_vehicle_states: Dict[str, Dict[str, object]] = {}
        self.match_score_fn = match_score_fn

    def update_scene(
        self,
        people: List[Detection],
        vehicles: List[Detection],
        frame,
        match_thresh: float,
    ) -> Tuple[Dict[int, str], Dict[int, List[TrackedDetection]], Dict[int, List[float]]]:
        self.frame_index += 1
        tracked_people = self._track_objects("person", self.person_tracker, people, frame)
        tracked_vehicles = self._track_objects("vehicle", self.vehicle_tracker, vehicles, frame)
        vehicle_matches = {idx: obj.stable_id for idx, obj in tracked_vehicles.items()}
        for vehicle_idx in range(len(vehicles)):
            vehicle_matches.setdefault(vehicle_idx, f"U{vehicle_idx + 1:03d}")

        current_pairs = self._score_current_pairs(tracked_people, tracked_vehicles, match_thresh)
        self._update_bindings(current_pairs)

        grouped: Dict[int, List[TrackedDetection]] = {idx: [] for idx in range(len(vehicles))}
        scores: Dict[int, List[float]] = {idx: [] for idx in range(len(vehicles))}
        vehicle_idx_by_stable = {obj.stable_id: idx for idx, obj in tracked_vehicles.items()}
        person_by_stable = {obj.stable_id: obj for obj in tracked_people.values()}

        for (person_id, vehicle_id), binding in self.bindings.items():
            if binding["state"] not in {"bound", "locked"}:
                continue
            if vehicle_id not in vehicle_idx_by_stable or person_id not in person_by_stable:
                continue
            vehicle_idx = vehicle_idx_by_stable[vehicle_id]
            grouped[vehicle_idx].append(person_by_stable[person_id])
            scores[vehicle_idx].append(float(binding["last_score"]))

        instant_grouped, instant_scores = self._match_raw_people_to_vehicles(people, vehicles, min(match_thresh, 0.95))
        self._update_vehicle_overload_states(instant_grouped, instant_scores, vehicle_matches, vehicles)
        return vehicle_matches, grouped, scores

    def _track_objects(self, kind: str, tracker, detections: List[Detection], frame) -> Dict[int, TrackedDetection]:
        if not detections:
            return {}
        det_array = np.array(
            [
                [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.confidence, det.class_id]
                for det in detections
            ],
            dtype=np.float32,
        )
        output = tracker.update(det_array, frame)
        raw_matches = self._match_outputs_to_detections(output, detections)
        tracked: Dict[int, TrackedDetection] = {}
        used_stable_ids: set[str] = set()
        for det_idx, raw_track_id in raw_matches.items():
            det = detections[det_idx]
            stable_id = self._stable_id_for(kind, raw_track_id, det, used_stable_ids)
            used_stable_ids.add(stable_id)
            tracked_det = TrackedDetection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=det.bbox,
                raw_track_id=raw_track_id,
                stable_id=stable_id,
            )
            for attr in ("has_helmet", "helmet_confidence", "source_class_name"):
                if hasattr(det, attr):
                    setattr(tracked_det, attr, getattr(det, attr))
            tracked[det_idx] = tracked_det
        return tracked

    def _match_outputs_to_detections(self, tracks_output, detections: List[Detection]) -> Dict[int, int]:
        if tracks_output is None or len(tracks_output) == 0:
            return {}
        candidates: List[Tuple[float, int, int]] = []
        for row in tracks_output:
            if len(row) < 5:
                continue
            x1, y1, x2, y2, track_id = row[:5]
            class_id = int(row[6]) if len(row) > 6 else None
            track_box = [float(x1), float(y1), float(x2), float(y2)]
            for det_idx, det in enumerate(detections):
                if class_id is not None and class_id != det.class_id:
                    continue
                score = iou(track_box, det.bbox)
                if score > 0.05:
                    candidates.append((score, int(track_id), det_idx))
        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matches: Dict[int, int] = {}
        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            matches[det_idx] = track_id
        return matches

    def _stable_id_for(
        self,
        kind: str,
        raw_track_id: int,
        det: Detection,
        used_stable_ids: Optional[set[str]] = None,
    ) -> str:
        used_stable_ids = used_stable_ids or set()
        mapped = self.raw_to_stable[kind].get(raw_track_id)
        if mapped and mapped not in used_stable_ids:
            self._touch_stable(mapped, det)
            return mapped
        if mapped in used_stable_ids:
            self.raw_to_stable[kind].pop(raw_track_id, None)

        prefix = "P" if kind == "person" else "M"
        candidate_id = self._find_reusable_stable_id(kind, det, used_stable_ids)
        if candidate_id is None:
            if kind == "person":
                candidate_id = f"{prefix}{self.next_person_id:03d}"
                self.next_person_id += 1
            else:
                candidate_id = f"{prefix}{self.next_vehicle_id:03d}"
                self.next_vehicle_id += 1
        self.raw_to_stable[kind][raw_track_id] = candidate_id
        self._touch_stable(candidate_id, det, kind=kind)
        return candidate_id

    def _touch_stable(self, stable_id: str, det: Detection, kind: Optional[str] = None) -> None:
        prev = self.stable_objects.get(stable_id)
        center = det.center
        velocity = (0.0, 0.0)
        if prev is not None:
            px, py = prev["center"]
            velocity = (center[0] - float(px), center[1] - float(py))
        self.stable_objects[stable_id] = {
            "kind": kind or (prev["kind"] if prev else ""),
            "bbox": det.bbox,
            "center": center,
            "confidence": det.confidence,
            "velocity": velocity,
            "last_seen": self.frame_index,
        }

    def _find_reusable_stable_id(
        self,
        kind: str,
        det: Detection,
        used_stable_ids: Optional[set[str]] = None,
    ) -> Optional[str]:
        used_stable_ids = used_stable_ids or set()
        best_score = 0.0
        best_id = None
        for stable_id, state in self.stable_objects.items():
            if stable_id in used_stable_ids:
                continue
            if state["kind"] != kind:
                continue
            missed = self.frame_index - int(state["last_seen"])
            if missed > self.max_missed + self.association_unbind_frames:
                continue
            old_box = state["bbox"]
            score = iou(old_box, det.bbox)
            if score > best_score:
                best_score = score
                best_id = stable_id
        return best_id if best_score >= 0.25 else None

    def _score_current_pairs(
        self,
        tracked_people: Dict[int, TrackedDetection],
        tracked_vehicles: Dict[int, TrackedDetection],
        match_thresh: float,
    ) -> Dict[Tuple[str, str], float]:
        candidates: List[Tuple[float, str, str]] = []
        for person in tracked_people.values():
            current_binding = self._active_vehicle_for_person(person.stable_id)
            for vehicle in tracked_vehicles.values():
                score = self.match_score_fn(person, vehicle)
                if score <= 0:
                    continue
                score += self._motion_score(person.stable_id, vehicle.stable_id)
                if (person.stable_id, vehicle.stable_id) in self.bindings:
                    score += 0.25
                if current_binding and current_binding != vehicle.stable_id:
                    current_score = float(self.bindings[(person.stable_id, current_binding)]["last_score"])
                    lock_until = int(self.bindings[(person.stable_id, current_binding)]["lock_until"])
                    if self.frame_index <= lock_until:
                        score -= 1.0
                    elif score < current_score + self.association_switch_margin:
                        score -= 0.5
                if score >= match_thresh:
                    candidates.append((score, person.stable_id, vehicle.stable_id))

        candidates.sort(reverse=True)
        used_people: set[str] = set()
        selected: Dict[Tuple[str, str], float] = {}
        for score, person_id, vehicle_id in candidates:
            if person_id in used_people:
                continue
            used_people.add(person_id)
            selected[(person_id, vehicle_id)] = score
        return selected

    def _match_raw_people_to_vehicles(
        self,
        people: List[Detection],
        vehicles: List[Detection],
        match_thresh: float,
    ) -> Tuple[Dict[int, List[Detection]], Dict[int, List[float]]]:
        grouped: Dict[int, List[Detection]] = {idx: [] for idx in range(len(vehicles))}
        scores: Dict[int, List[float]] = {idx: [] for idx in range(len(vehicles))}
        candidates: List[Tuple[float, int, int]] = []
        for person_idx, person in enumerate(people):
            for vehicle_idx, vehicle in enumerate(vehicles):
                score = self.match_score_fn(person, vehicle)
                if score >= match_thresh:
                    candidates.append((score, person_idx, vehicle_idx))
        candidates.sort(reverse=True)

        used_people: set[int] = set()
        for score, person_idx, vehicle_idx in candidates:
            if person_idx in used_people:
                continue
            used_people.add(person_idx)
            grouped[vehicle_idx].append(people[person_idx])
            scores[vehicle_idx].append(score)
        return grouped, scores

    def _motion_score(self, person_id: str, vehicle_id: str) -> float:
        person = self.stable_objects.get(person_id)
        vehicle = self.stable_objects.get(vehicle_id)
        if not person or not vehicle:
            return 0.0
        pv = person["velocity"]
        vv = vehicle["velocity"]
        pnorm = (float(pv[0]) ** 2 + float(pv[1]) ** 2) ** 0.5
        vnorm = (float(vv[0]) ** 2 + float(vv[1]) ** 2) ** 0.5
        if pnorm < 1.0 or vnorm < 1.0:
            return 0.0
        cosine = (float(pv[0]) * float(vv[0]) + float(pv[1]) * float(vv[1])) / (pnorm * vnorm)
        return max(0.0, cosine) * 0.15

    def _active_vehicle_for_person(self, person_id: str) -> Optional[str]:
        active = []
        for (bound_person_id, vehicle_id), binding in self.bindings.items():
            if bound_person_id != person_id:
                continue
            if binding["state"] in {"bound", "locked", "lost"}:
                active.append((int(binding["hits"]), vehicle_id))
        if not active:
            return None
        active.sort(reverse=True)
        return active[0][1]

    def _update_bindings(self, current_pairs: Dict[Tuple[str, str], float]) -> None:
        for key, score in current_pairs.items():
            binding = self.bindings.setdefault(
                key,
                {
                    "hits": 0,
                    "misses": 0,
                    "state": "candidate",
                    "lock_until": 0,
                    "last_score": 0.0,
                },
            )
            binding["hits"] = int(binding["hits"]) + 1
            binding["misses"] = 0
            binding["last_score"] = score
            if int(binding["hits"]) >= self.association_min_hits:
                binding["state"] = "locked" if self.frame_index <= int(binding["lock_until"]) else "bound"
                if int(binding["lock_until"]) < self.frame_index:
                    binding["lock_until"] = self.frame_index + self.association_lock_frames
                    binding["state"] = "locked"

        expired = []
        for key, binding in self.bindings.items():
            if key in current_pairs:
                continue
            binding["misses"] = int(binding["misses"]) + 1
            if binding["state"] in {"bound", "locked"}:
                binding["state"] = "lost"
            if int(binding["misses"]) > self.association_unbind_frames:
                expired.append(key)
        for key in expired:
            self.bindings.pop(key, None)

    def _update_vehicle_overload_states(
        self,
        instant_grouped: Dict[int, List[Detection]],
        instant_scores: Dict[int, List[float]],
        vehicle_matches: Dict[int, str],
        vehicles: List[Detection],
    ) -> None:
        current_vehicle_ids = set(vehicle_matches.values())
        for vehicle_idx, stable_id in vehicle_matches.items():
            state = self.vehicle_states.setdefault(stable_id, self._new_vehicle_state())
            evidence_people = instant_grouped.get(vehicle_idx, [])
            evidence_scores = instant_scores.get(vehicle_idx, [])
            count = len(evidence_people)
            stable_obj = self.stable_objects.get(stable_id, {})
            velocity = stable_obj.get("velocity", (0.0, 0.0))
            speed_px = (float(velocity[0]) ** 2 + float(velocity[1]) ** 2) ** 0.5
            state["last_seen"] = self.frame_index
            vehicle_confidence = vehicles[vehicle_idx].confidence if vehicle_idx < len(vehicles) else float(stable_obj.get("confidence", 0.0))
            update_simple_overload_stats(
                state,
                rider_count=count,
                vehicle_confidence=vehicle_confidence,
                matched_people=evidence_people,
                match_scores=evidence_scores,
                speed_px=speed_px,
            )
            if state["confirmed_overload"]:
                self.confirmed_track_ids.add(stable_id)

        for stable_id, state in list(self.vehicle_states.items()):
            if stable_id in current_vehicle_ids:
                continue
            if self.frame_index - int(state["last_seen"]) > self.max_missed:
                self._archive_vehicle_state(stable_id, state)
                self.vehicle_states.pop(stable_id, None)
            else:
                state["confirmed_overload"] = False
                state["overload_status"] = "NORMAL"

    def is_confirmed(self, track_id: int | str) -> bool:
        state = self.vehicle_states.get(str(track_id))
        return bool(state and state["confirmed_overload"])

    def get_overload_status(self, track_id: int | str) -> str:
        track_id = str(track_id)
        if track_id.startswith("U"):
            return "UNCERTAIN"
        state = self.vehicle_states.get(track_id)
        if state is None:
            state = self.archived_vehicle_states.get(track_id)
        if not state:
            return "NORMAL"
        return str(state.get("overload_status") or overload_status_from_stats(state))

    def get_track_stats(self) -> Dict[str, Dict[str, object]]:
        stats = {
            str(track_id): self._merge_vehicle_states(None, state)
            for track_id, state in self.archived_vehicle_states.items()
        }
        for track_id, state in self.vehicle_states.items():
            if track_id in stats:
                stats[track_id] = self._merge_vehicle_states(stats[track_id], state)
            else:
                stats[track_id] = self._merge_vehicle_states(None, state)
        return stats

    def _new_vehicle_state(self) -> Dict[str, object]:
        return {
            "observed_frames": 0,
            "raw_overload_frames": 0,
            "high_conf_observed_frames": 0,
            "high_conf_overload_frames": 0,
            "moving_observed_frames": 0,
            "moving_overload_frames": 0,
            "max_rider_count": 0,
            "last_rider_count": 0,
            "sum_vehicle_conf": 0.0,
            "sum_match_score": 0.0,
            "sum_speed_px": 0.0,
            "raw_overload_ratio": 0.0,
            "high_conf_overload_ratio": 0.0,
            "moving_overload_ratio": 0.0,
            "avg_vehicle_conf": 0.0,
            "avg_match_score": 0.0,
            "avg_speed_px": 0.0,
            "last_seen": self.frame_index,
            "overload_status": "NORMAL",
            "suspected_overload": False,
            "confirmed_overload": False,
            "ever_confirmed_overload": False,
        }

    def _archive_vehicle_state(self, stable_id: str, state: Dict[str, object]) -> None:
        previous = self.archived_vehicle_states.get(stable_id)
        self.archived_vehicle_states[stable_id] = (
            self._merge_vehicle_states(previous, state) if previous else dict(state)
        )

    def _merge_vehicle_states(
        self,
        first: Optional[Dict[str, object]],
        second: Dict[str, object],
    ) -> Dict[str, object]:
        if first is None:
            merged = dict(second)
        else:
            merged = dict(first)
            summed_keys = (
                "observed_frames",
                "raw_overload_frames",
                "high_conf_observed_frames",
                "high_conf_overload_frames",
                "moving_observed_frames",
                "moving_overload_frames",
                "sum_vehicle_conf",
                "sum_match_score",
                "sum_speed_px",
            )
            for key in summed_keys:
                merged[key] = float(merged.get(key, 0.0) or 0.0) + float(second.get(key, 0.0) or 0.0)
            int_keys = (
                "observed_frames",
                "raw_overload_frames",
                "high_conf_observed_frames",
                "high_conf_overload_frames",
                "moving_observed_frames",
                "moving_overload_frames",
            )
            for key in int_keys:
                merged[key] = int(merged[key])
            merged["max_rider_count"] = max(int(first.get("max_rider_count", 0) or 0), int(second.get("max_rider_count", 0) or 0))
            merged["last_rider_count"] = int(second.get("last_rider_count", first.get("last_rider_count", 0)) or 0)
            merged["last_seen"] = max(int(first.get("last_seen", 0) or 0), int(second.get("last_seen", 0) or 0))
            merged["ever_confirmed_overload"] = bool(first.get("ever_confirmed_overload", False)) or bool(second.get("ever_confirmed_overload", False))

        observed = int(merged.get("observed_frames", 0) or 0)
        high_conf_observed = int(merged.get("high_conf_observed_frames", 0) or 0)
        moving_observed = int(merged.get("moving_observed_frames", 0) or 0)
        raw_frames = int(merged.get("raw_overload_frames", 0) or 0)
        high_conf_frames = int(merged.get("high_conf_overload_frames", 0) or 0)
        moving_frames = int(merged.get("moving_overload_frames", 0) or 0)
        merged["raw_overload_ratio"] = raw_frames / observed if observed else 0.0
        merged["high_conf_overload_ratio"] = high_conf_frames / high_conf_observed if high_conf_observed else 0.0
        merged["moving_overload_ratio"] = moving_frames / moving_observed if moving_observed else 0.0
        merged["avg_vehicle_conf"] = float(merged.get("sum_vehicle_conf", 0.0) or 0.0) / observed if observed else 0.0
        merged["avg_match_score"] = float(merged.get("sum_match_score", 0.0) or 0.0) / observed if observed else 0.0
        merged["avg_speed_px"] = float(merged.get("sum_speed_px", 0.0) or 0.0) / observed if observed else 0.0
        status = overload_status_from_stats(merged)
        merged["overload_status"] = status
        merged["suspected_overload"] = status in {"SUSPECTED", "CONFIRMED"}
        merged["confirmed_overload"] = status == "CONFIRMED"
        merged["ever_confirmed_overload"] = bool(merged.get("ever_confirmed_overload", False)) or status == "CONFIRMED"
        return merged


class AssociationByteTracker(AssociationMotTracker):
    def __init__(
        self,
        track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        confirm_frames: int,
        max_missed: int,
        association_min_hits: int,
        association_lock_frames: int,
        association_unbind_frames: int,
        association_switch_margin: float,
        match_score_fn: Callable[[Detection, Detection], float] = person_vehicle_match_score,
    ):
        try:
            from boxmot import BYTETracker
        except ImportError as exc:
            raise RuntimeError("Association ByteTrack requires boxmot. Install project tracking dependencies first.") from exc

        person_tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=30,
        )
        vehicle_tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=30,
        )
        super().__init__(
            person_tracker=person_tracker,
            vehicle_tracker=vehicle_tracker,
            confirm_frames=confirm_frames,
            max_missed=max_missed,
            association_min_hits=association_min_hits,
            association_lock_frames=association_lock_frames,
            association_unbind_frames=association_unbind_frames,
            association_switch_margin=association_switch_margin,
            match_score_fn=match_score_fn,
        )
