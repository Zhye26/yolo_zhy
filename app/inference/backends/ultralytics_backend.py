"""
Ultralytics YOLO backend implementation.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from app.inference.backends.base import DetectorBackend
from app.core.types import Detection
from app.config import settings


class UltralyticsBackend(DetectorBackend):
    """Ultralytics YOLO backend for inference."""

    def __init__(self):
        self._model = None
        self._class_names: List[str] = []
        self._candidate_model = None
        self._candidate_names: Dict[int, str] = {}
        self._candidate_class_ids: set[int] = set()
        self._supports_candidate_refine = False
        self._temporal_memory: Dict[int, List[Dict[str, Any]]] = {}
        self._frame_index = 0

    def load(self, model_path: str) -> None:
        """Load Ultralytics YOLO model."""
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._class_names = list(self._model.names.values())
        self._supports_candidate_refine = self._class_names == settings.detection.class_names

        if self._supports_candidate_refine and settings.model.enable_candidate_refine:
            candidate_path = str(settings.model.candidate_model_path)
            self._candidate_model = YOLO(candidate_path)
            self._candidate_names = dict(self._candidate_model.names)
            self._candidate_class_ids = self._resolve_candidate_class_ids(self._candidate_names)

        self.reset()

    def reset(self) -> None:
        """Reset per-stream temporal state."""
        self._temporal_memory = {}
        self._frame_index = 0

    def _enabled_detection_class_ids(self) -> List[int]:
        class_ids = [
            settings.detection.ebike_class_id,
            settings.detection.driver_class_id,
            settings.detection.passenger_class_id,
        ]
        if settings.detection.helmet_detection_enabled:
            class_ids.append(settings.detection.helmet_class_id)
        return class_ids

    def predict(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45
    ) -> List[Detection]:
        """Run inference using Ultralytics."""
        if not self._model:
            return []

        self._frame_index += 1
        candidate_regions: List[Dict[str, Any]] = []

        observed_detections = self._filter_detections(
            self._predict_with_model(
                self._model,
                frame,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                imgsz=settings.model.imgsz,
                class_names=self._class_names,
                classes=self._enabled_detection_class_ids(),
            ),
            frame.shape,
        )

        should_refine = self._should_refine(observed_detections)
        if (should_refine or self._needs_context_prune(observed_detections)) and not candidate_regions:
            candidate_regions = self._find_candidate_regions(frame)

        if should_refine:
            refined = self._predict_with_candidate_regions(
                frame,
                candidate_regions=candidate_regions,
                iou_thresh=iou_thresh,
            )
            if settings.model.enable_tile_refine:
                tile_detections = self._predict_with_tiles(frame, iou_thresh=iou_thresh)
                refined.extend(
                    self._select_tile_detections(
                        tile_detections,
                        candidate_regions=candidate_regions,
                        observed_detections=observed_detections,
                        frame_shape=frame.shape,
                    )
                )
            observed_detections = self._merge_detections(
                observed_detections,
                self._filter_detections(refined, frame.shape),
                iou_thresh=iou_thresh,
            )

        final_detections = observed_detections
        if settings.model.enable_temporal_stabilize and self._should_refine(final_detections):
            if not candidate_regions:
                candidate_regions = self._find_candidate_regions(frame)
            temporal_detections = self._build_temporal_detections(
                observed_detections,
                candidate_regions=candidate_regions,
            )
            if temporal_detections:
                final_detections = self._merge_detections(
                    final_detections,
                    temporal_detections,
                    iou_thresh=iou_thresh,
                )

        if candidate_regions:
            proxy_detections = self._build_contextual_proxy_detections(
                final_detections,
                candidate_regions=candidate_regions,
                frame_shape=frame.shape,
            )
            if proxy_detections:
                final_detections = self._merge_detections(
                    final_detections,
                    proxy_detections,
                    iou_thresh=iou_thresh,
                )

            final_detections = self._prune_contextless_detections(
                final_detections,
                candidate_regions=candidate_regions,
            )

        final_detections = self._consolidate_ebike_detections(
            final_detections,
            candidate_regions=candidate_regions,
        )
        final_detections = self._consolidate_same_class_detections(final_detections)
        if settings.detection.helmet_detection_enabled:
            final_detections = self._consolidate_helmet_detections(
                final_detections,
                candidate_regions=candidate_regions,
            )
        self._update_temporal_memory(observed_detections)
        return final_detections

    def _predict_with_model(
        self,
        model,
        frame: np.ndarray,
        conf_thresh: float,
        iou_thresh: float,
        imgsz: int,
        class_names: List[str],
        classes: Optional[List[int]] = None,
    ) -> List[Detection]:
        results = model.predict(
            frame,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=imgsz,
            classes=classes,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes, confs, classes):
                class_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
                detections.append(Detection(
                    bbox=bbox.tolist(),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                ))

        return detections

    def _should_refine(self, detections: List[Detection]) -> bool:
        if not self._supports_candidate_refine:
            return False
        if self._candidate_model is None or not self._candidate_class_ids:
            return False
        if not detections:
            return True

        has_reliable_ebike = any(
            d.class_id == settings.detection.ebike_class_id
            and self._has_local_ebike_support(d, detections)
            for d in detections
        )
        has_reliable_driver = any(
            d.class_id == settings.detection.driver_class_id
            and self._has_local_driver_support(d, detections)
            for d in detections
        )
        return not has_reliable_ebike or not has_reliable_driver

    def _needs_context_prune(self, detections: List[Detection]) -> bool:
        for det in detections:
            if det.class_id == settings.detection.ebike_class_id and not self._has_local_ebike_support(det, detections):
                return True
            if det.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id}:
                if not self._has_local_driver_support(det, detections):
                    return True
        return False

    def _predict_with_candidate_regions(
        self,
        frame: np.ndarray,
        candidate_regions: List[Dict[str, Any]],
        iou_thresh: float,
    ) -> List[Detection]:
        refined_detections: List[Detection] = []

        for region in candidate_regions:
            crop, offset = self._crop_region(frame, region)
            if crop.size == 0:
                continue

            crop_detections = self._predict_with_model(
                self._model,
                crop,
                conf_thresh=settings.model.refine_conf_thresh,
                iou_thresh=iou_thresh,
                imgsz=settings.model.refine_imgsz,
                class_names=self._class_names,
                classes=self._enabled_detection_class_ids(),
            )
            restored = self._restore_crop_detections(crop_detections, offset, frame.shape)
            refined_detections.extend(restored)

            if region["class_name"] in {"bicycle", "motorcycle"}:
                refined_detections.extend(self._build_proxy_ebike_detections(region))

        return refined_detections

    def _predict_with_tiles(
        self,
        frame: np.ndarray,
        iou_thresh: float,
    ) -> List[Detection]:
        h, w = frame.shape[:2]
        tile_w = max(64, int(w * settings.model.tile_refine_width_ratio))
        tile_h = max(64, int(h * settings.model.tile_refine_height_ratio))
        stride_x = max(1, int(tile_w * (1 - settings.model.tile_refine_overlap)))
        stride_y = max(1, int(tile_h * (1 - settings.model.tile_refine_overlap)))

        offsets_x = self._build_offsets(length=w, tile_length=tile_w, stride=stride_x)
        offsets_y = self._build_offsets(length=h, tile_length=tile_h, stride=stride_y)

        detections: List[Detection] = []
        for offset_y in offsets_y:
            for offset_x in offsets_x:
                crop = frame[offset_y:min(h, offset_y + tile_h), offset_x:min(w, offset_x + tile_w)]
                crop_detections = self._predict_with_model(
                    self._model,
                    crop,
                    conf_thresh=settings.model.tile_refine_conf_thresh,
                    iou_thresh=iou_thresh,
                    imgsz=settings.model.tile_refine_imgsz,
                    class_names=self._class_names,
                    classes=self._enabled_detection_class_ids(),
                )
                detections.extend(
                    self._restore_crop_detections(crop_detections, (offset_x, offset_y), frame.shape)
                )

        return detections

    def _select_tile_detections(
        self,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
        observed_detections: List[Detection],
        frame_shape: Tuple[int, ...],
    ) -> List[Detection]:
        ebike_class_id = settings.detection.ebike_class_id
        driver_class_id = settings.detection.driver_class_id
        passenger_class_id = settings.detection.passenger_class_id
        helmet_class_id = settings.detection.helmet_class_id

        selected: List[Detection] = []
        filtered = self._filter_detections(detections, frame_shape)

        for det in sorted(filtered, key=lambda item: item.confidence, reverse=True):
            if det.class_id == helmet_class_id:
                if det.confidence >= 0.05:
                    selected.append(det)
                continue

            if det.class_id == ebike_class_id:
                if det.confidence >= 0.08 or self._has_ebike_support(det, candidate_regions, observed_detections):
                    selected.append(self._boost_detection(det, floor=0.08, ceiling=0.35))
                continue

            if det.class_id == driver_class_id:
                if det.confidence >= 0.04 and self._has_driver_support(det, candidate_regions, observed_detections):
                    selected.append(self._boost_detection(det, floor=0.12, ceiling=0.32))
                continue

            if det.class_id == passenger_class_id:
                if det.confidence >= 0.04 and self._has_driver_support(det, candidate_regions, observed_detections):
                    selected.append(self._boost_detection(det, floor=0.12, ceiling=0.34))

        return selected

    def _build_contextual_proxy_detections(
        self,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
        frame_shape: Tuple[int, ...],
    ) -> List[Detection]:
        ebike_class_id = settings.detection.ebike_class_id
        driver_class_id = settings.detection.driver_class_id
        passenger_class_id = settings.detection.passenger_class_id
        helmet_class_id = settings.detection.helmet_class_id

        ebikes = [det for det in detections if det.class_id == ebike_class_id]
        drivers = [det for det in detections if det.class_id == driver_class_id]
        passengers = [det for det in detections if det.class_id == passenger_class_id]
        helmets = [det for det in detections if det.class_id == helmet_class_id]
        person_regions = [region for region in candidate_regions if region["class_name"] == "person"]

        proxies: List[Detection] = []
        for ebike in ebikes:
            matched_riders = [
                rider for rider in drivers + passengers
                if self._person_matches_ebike(rider.bbox, ebike.bbox)
            ]
            related_regions = self._find_related_person_regions(ebike.bbox, person_regions)

            proxies.extend(
                self._build_rider_proxy_detections(
                    ebike=ebike,
                    matched_riders=matched_riders,
                    related_regions=related_regions,
                    helmets=helmets,
                    frame_shape=frame_shape,
                    rider_class_ids=(driver_class_id, passenger_class_id),
                )
            )

        return self._filter_detections(proxies, frame_shape)

    def _build_rider_proxy_detections(
        self,
        ebike: Detection,
        matched_riders: List[Detection],
        related_regions: List[Dict[str, Any]],
        helmets: List[Detection],
        frame_shape: Tuple[int, ...],
        rider_class_ids: Tuple[int, int],
    ) -> List[Detection]:
        driver_class_id, passenger_class_id = rider_class_ids
        proxies: List[Detection] = []
        rider_count = len(matched_riders)
        target_count = min(2, max(rider_count, len(related_regions)))

        for region in related_regions:
            if rider_count >= target_count:
                break
            if not self._is_viable_rider_proxy_region(
                region=region,
                ebike_bbox=ebike.bbox,
                existing_riders=matched_riders + [proxy for proxy in proxies],
                is_additional_rider=rider_count > 0,
            ):
                continue
            if self._region_matches_any_rider(
                region["bbox"],
                matched_riders + [proxy for proxy in proxies],
            ):
                continue

            class_id = driver_class_id if rider_count == 0 else passenger_class_id
            proxies.append(self._build_proxy_person_detection(region, ebike, class_id))
            rider_count += 1

        if rider_count == 0:
            helmet = self._find_related_helmet(ebike.bbox, helmets)
            if helmet is not None:
                proxies.append(
                    Detection(
                        class_id=driver_class_id,
                        confidence=min(max(max(helmet.confidence, ebike.confidence) * 0.55, 0.10), 0.28),
                        class_name=self._class_names[driver_class_id],
                        bbox=self._estimate_driver_bbox_from_helmet(helmet.bbox, frame_shape),
                    )
                )

        return proxies

    def _is_viable_rider_proxy_region(
        self,
        region: Dict[str, Any],
        ebike_bbox: List[float],
        existing_riders: List[Detection],
        is_additional_rider: bool,
    ) -> bool:
        bbox = region["bbox"]
        support = self._person_ebike_overlap_score(bbox, ebike_bbox)
        x1, y1, x2, y2 = bbox
        ex1, ey1, ex2, ey2 = ebike_bbox
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        ebike_w = max(1.0, ex2 - ex1)
        ebike_h = max(1.0, ey2 - ey1)
        center_x = (x1 + x2) / 2
        bottom_y = y2

        if box_w < 14 or box_h < 26:
            return False
        if region["confidence"] < (0.28 if is_additional_rider else 0.18) and support < (0.70 if is_additional_rider else 0.58):
            return False
        if not (ex1 - ebike_w * 0.10 <= center_x <= ex2 + ebike_w * 0.10):
            return False
        if not (ey1 - ebike_h * 0.12 <= bottom_y <= ey2 + ebike_h * 0.10):
            return False

        if is_additional_rider and not self._is_separated_additional_rider(bbox, existing_riders, ebike_bbox):
            return False

        return True

    def _is_separated_additional_rider(
        self,
        region_bbox: List[float],
        riders: List[Detection],
        ebike_bbox: List[float],
    ) -> bool:
        if not riders:
            return True

        ebike_center_x = (ebike_bbox[0] + ebike_bbox[2]) / 2
        region_center_x = (region_bbox[0] + region_bbox[2]) / 2
        min_center_gap = max(18.0, (ebike_bbox[2] - ebike_bbox[0]) * 0.12)

        for rider in riders:
            rider_center_x = (rider.bbox[0] + rider.bbox[2]) / 2
            if abs(region_center_x - rider_center_x) < min_center_gap:
                return False
            if abs(region_center_x - ebike_center_x) < 6.0 and abs(rider_center_x - ebike_center_x) < 6.0:
                return False

        return True

    def _prune_contextless_detections(
        self,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> List[Detection]:
        kept: List[Detection] = []

        for det in detections:
            if det.class_id == settings.detection.ebike_class_id:
                if det.confidence >= 0.45 or self._has_ebike_support(det, candidate_regions, detections):
                    kept.append(det)
                continue

            if det.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id}:
                if det.confidence >= 0.08 and self._has_driver_support(det, candidate_regions, detections):
                    kept.append(det)
                continue

            kept.append(det)

        return kept

    def _consolidate_ebike_detections(
        self,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> List[Detection]:
        ebike_class_id = settings.detection.ebike_class_id
        ebikes = [det for det in detections if det.class_id == ebike_class_id]
        if len(ebikes) < 2:
            return detections

        grouped: Dict[str, List[Detection]] = {}
        ungrouped: List[Detection] = []

        for det in ebikes:
            group_key = self._resolve_ebike_group_key(det, detections, candidate_regions)
            if group_key is None:
                ungrouped.append(det)
                continue
            grouped.setdefault(group_key, []).append(det)

        kept_ebikes: List[Detection] = []
        for group in grouped.values():
            anchor = max(
                group,
                key=lambda item: self._score_ebike_detection(item, detections, candidate_regions),
            )
            kept_ebikes.append(
                self._expand_ebike_detection(
                    anchor,
                    sibling_ebikes=group,
                    detections=detections,
                    candidate_regions=candidate_regions,
                )
            )

        for det in sorted(ungrouped, key=lambda item: item.confidence, reverse=True):
            if any(
                self._iou(det.bbox, existing.bbox) > 0.25
                or self._boxes_related(det.bbox, existing.bbox, expand_ratio=0.15)
                for existing in kept_ebikes
            ):
                continue
            kept_ebikes.append(
                self._expand_ebike_detection(
                    det,
                    sibling_ebikes=[det],
                    detections=detections,
                    candidate_regions=candidate_regions,
                )
            )

        non_ebikes = [det for det in detections if det.class_id != ebike_class_id]
        return sorted(non_ebikes + kept_ebikes, key=lambda item: item.confidence, reverse=True)

    def _resolve_ebike_group_key(
        self,
        detection: Detection,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> Optional[str]:
        rider_group = self._find_best_rider_group(detection, detections, candidate_regions)
        if rider_group is not None:
            return rider_group

        bike_group = self._find_best_bike_group(detection, candidate_regions)
        if bike_group is not None:
            return bike_group

        return None

    def _consolidate_same_class_detections(
        self,
        detections: List[Detection],
    ) -> List[Detection]:
        grouped: Dict[int, List[Detection]] = {}
        for det in detections:
            grouped.setdefault(det.class_id, []).append(det)

        kept: List[Detection] = []
        for class_id, group in grouped.items():
            ordered = sorted(
                group,
                key=lambda item: self._duplicate_keep_score(item),
                reverse=True,
            )
            class_kept: List[Detection] = []
            for det in ordered:
                if any(self._is_same_class_duplicate(det, existing) for existing in class_kept):
                    continue
                class_kept.append(det)
            kept.extend(class_kept)

        return sorted(kept, key=lambda item: item.confidence, reverse=True)

    def _duplicate_keep_score(self, detection: Detection) -> float:
        area_bonus = min(detection.area / 50000.0, 0.2)
        return detection.confidence + area_bonus

    def _is_same_class_duplicate(
        self,
        candidate: Detection,
        existing: Detection,
    ) -> bool:
        if candidate.class_id != existing.class_id:
            return False

        overlap = self._overlap_ratio(candidate.bbox, existing.bbox)
        iou = self._iou(candidate.bbox, existing.bbox)
        center_distance = self._center_distance(candidate.bbox, existing.bbox)
        min_dim = min(
            candidate.bbox[2] - candidate.bbox[0],
            candidate.bbox[3] - candidate.bbox[1],
            existing.bbox[2] - existing.bbox[0],
            existing.bbox[3] - existing.bbox[1],
        )

        if candidate.class_id == settings.detection.helmet_class_id:
            return overlap > 0.50 or (iou > 0.12 and center_distance < max(10.0, min_dim * 0.9))

        if candidate.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id}:
            return overlap > 0.58 or (iou > 0.22 and center_distance < max(18.0, min_dim * 0.8))

        if candidate.class_id == settings.detection.ebike_class_id:
            smaller_in_larger = (
                min(candidate.area, existing.area) / max(candidate.area, existing.area) < 0.45
                and (overlap > 0.80 or self._box_contains(candidate.bbox, existing.bbox, margin=0.08) or self._box_contains(existing.bbox, candidate.bbox, margin=0.08))
            )
            return smaller_in_larger or overlap > 0.62 or (iou > 0.24 and center_distance < max(24.0, min_dim * 1.4))

        return overlap > 0.65 or iou > 0.4


    def _find_best_rider_group(
        self,
        detection: Detection,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> Optional[str]:
        best_candidate_key: Optional[str] = None
        best_candidate_score = 0.0
        for index, region in enumerate(candidate_regions):
            if region["class_name"] != "person":
                continue
            if not self._person_matches_ebike(region["bbox"], detection.bbox):
                continue
            score = self._person_ebike_overlap_score(region["bbox"], detection.bbox)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate_key = f"cand-person:{index}"

        if best_candidate_key is not None:
            return best_candidate_key

        rider_class_ids = {settings.detection.driver_class_id, settings.detection.passenger_class_id}
        best_key: Optional[str] = None
        best_score = 0.0
        for index, det in enumerate(detections):
            if det.class_id not in rider_class_ids:
                continue
            if not self._person_matches_ebike(det.bbox, detection.bbox):
                continue
            score = self._person_ebike_overlap_score(det.bbox, detection.bbox)
            if score > best_score:
                best_score = score
                best_key = f"det-person:{index}"

        return best_key

    def _consolidate_helmet_detections(
        self,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> List[Detection]:
        if not settings.detection.helmet_detection_enabled:
            return detections
        helmet_class_id = settings.detection.helmet_class_id
        helmets = [det for det in detections if det.class_id == helmet_class_id]
        if len(helmets) < 2:
            return detections

        grouped: Dict[str, Detection] = {}
        ungrouped: List[Detection] = []
        for helmet in sorted(helmets, key=lambda item: item.confidence, reverse=True):
            group_key = self._resolve_helmet_group_key(helmet, detections, candidate_regions)
            if group_key is None:
                if helmet.confidence >= 0.18:
                    ungrouped.append(helmet)
                continue

            existing = grouped.get(group_key)
            if existing is None or self._score_helmet_detection(helmet, detections, candidate_regions) > self._score_helmet_detection(existing, detections, candidate_regions):
                grouped[group_key] = helmet

        kept_helmets = list(grouped.values())
        for helmet in ungrouped:
            if any(self._is_same_class_duplicate(helmet, existing) for existing in kept_helmets):
                continue
            kept_helmets.append(helmet)

        others = [det for det in detections if det.class_id != helmet_class_id]
        return sorted(others + kept_helmets, key=lambda item: item.confidence, reverse=True)

    def _resolve_helmet_group_key(
        self,
        helmet: Detection,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> Optional[str]:
        rider_class_ids = {settings.detection.driver_class_id, settings.detection.passenger_class_id}
        best_key: Optional[str] = None
        best_score = 0.0

        for index, det in enumerate(detections):
            if det.class_id not in rider_class_ids:
                continue
            score = self._helmet_person_score(helmet.bbox, det.bbox)
            if score > best_score:
                best_score = score
                best_key = f"det-person:{index}"

        for index, region in enumerate(candidate_regions):
            if region["class_name"] != "person":
                continue
            score = self._helmet_person_score(helmet.bbox, region["bbox"])
            if score > best_score:
                best_score = score
                best_key = f"cand-person:{index}"

        return best_key if best_score >= 0.22 else None

    def _score_helmet_detection(
        self,
        helmet: Detection,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> float:
        score = helmet.confidence
        rider_key = self._resolve_helmet_group_key(helmet, detections, candidate_regions)
        if rider_key is None:
            return score

        if rider_key.startswith('det-person:'):
            det_index = int(rider_key.split(':', 1)[1])
            score += self._helmet_person_score(helmet.bbox, detections[det_index].bbox)
        else:
            region_index = int(rider_key.split(':', 1)[1])
            score += self._helmet_person_score(helmet.bbox, candidate_regions[region_index]['bbox'])
        return score

    def _helmet_person_score(
        self,
        helmet_bbox: List[float],
        person_bbox: List[float],
    ) -> float:
        px1, py1, px2, py2 = person_bbox
        person_w = px2 - px1
        person_h = py2 - py1
        head_box = [
            px1 - person_w * 0.12,
            py1 - person_h * 0.08,
            px2 + person_w * 0.12,
            py1 + person_h * 0.42,
        ]
        score = self._overlap_ratio(helmet_bbox, head_box)
        helmet_center_x = (helmet_bbox[0] + helmet_bbox[2]) / 2
        helmet_center_y = (helmet_bbox[1] + helmet_bbox[3]) / 2
        if head_box[0] <= helmet_center_x <= head_box[2] and head_box[1] <= helmet_center_y <= head_box[3]:
            score += 0.45
        if helmet_center_y <= py1 + person_h * 0.48:
            score += 0.12
        return score

    def _find_best_bike_group(
        self,
        detection: Detection,
        candidate_regions: List[Dict[str, Any]],
    ) -> Optional[str]:
        best_key: Optional[str] = None
        best_score = 0.0

        for index, region in enumerate(candidate_regions):
            if region["class_name"] not in {"bicycle", "motorcycle"}:
                continue
            score = self._bike_region_support_score(detection.bbox, region["bbox"])
            if score > best_score:
                best_score = score
                best_key = f"cand-bike:{index}"

        return best_key

    def _score_ebike_detection(
        self,
        detection: Detection,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> float:
        score = detection.confidence
        if self._has_local_ebike_support(detection, detections):
            score += 0.35

        best_bike_support = 0.0
        for region in candidate_regions:
            if region["class_name"] in {"bicycle", "motorcycle"}:
                best_bike_support = max(
                    best_bike_support,
                    self._bike_region_support_score(detection.bbox, region["bbox"]),
                )
        score += best_bike_support * 0.35

        best_rider_support = 0.0
        for det in detections:
            if det.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id}:
                best_rider_support = max(
                    best_rider_support,
                    self._person_ebike_overlap_score(det.bbox, detection.bbox),
                )
        for region in candidate_regions:
            if region["class_name"] == "person":
                best_rider_support = max(
                    best_rider_support,
                    self._person_ebike_overlap_score(region["bbox"], detection.bbox),
                )
        score += best_rider_support * 0.25
        score += min((detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]) / 60000.0, 0.15)
        return score

    def _expand_ebike_detection(
        self,
        detection: Detection,
        sibling_ebikes: List[Detection],
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> Detection:
        merged_boxes: List[List[float]] = [detection.bbox]

        for sibling in sibling_ebikes:
            if sibling is detection:
                continue
            if (
                self._iou(detection.bbox, sibling.bbox) > 0.08
                or self._boxes_related(detection.bbox, sibling.bbox, expand_ratio=0.18)
            ):
                merged_boxes.append(sibling.bbox)

        supporting_region = self._find_best_supporting_bike_region(
            detection,
            detections=detections,
            candidate_regions=candidate_regions,
        )
        if supporting_region is not None:
            merged_boxes.append(supporting_region["bbox"])

        expanded_bbox = self._limit_ebike_bbox_growth(
            detection.bbox,
            self._merge_bboxes(merged_boxes),
        )
        return Detection(
            class_id=detection.class_id,
            confidence=detection.confidence,
            class_name=detection.class_name,
            bbox=expanded_bbox,
        )

    def _find_best_supporting_bike_region(
        self,
        detection: Detection,
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        related_people = self._collect_related_person_boxes(detection.bbox, detections, candidate_regions)
        best_region: Optional[Dict[str, Any]] = None
        best_score = 0.0
        anchor_area = max(detection.area, 1.0)

        for region in candidate_regions:
            if region["class_name"] not in {"bicycle", "motorcycle"}:
                continue

            support = self._bike_region_support_score(detection.bbox, region["bbox"])
            if support < 0.18 and not self._boxes_related(detection.bbox, region["bbox"], expand_ratio=0.10):
                continue

            region_area = max(
                1.0,
                (region["bbox"][2] - region["bbox"][0]) * (region["bbox"][3] - region["bbox"][1]),
            )
            if region_area > anchor_area * 3.0 and support < 0.45:
                continue

            if related_people and not any(
                self._person_matches_ebike(person_bbox, region["bbox"])
                for person_bbox in related_people
            ):
                continue

            score = support + min(region["confidence"], 0.65) * 0.15
            if region_area > anchor_area:
                score += min(region_area / anchor_area, 2.0) * 0.08

            if score > best_score:
                best_score = score
                best_region = region

        return best_region

    def _collect_related_person_boxes(
        self,
        ebike_bbox: List[float],
        detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> List[List[float]]:
        related_boxes: List[List[float]] = []

        for det in detections:
            if det.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id}:
                if self._person_matches_ebike(det.bbox, ebike_bbox):
                    related_boxes.append(det.bbox)

        for region in candidate_regions:
            if region["class_name"] == "person" and self._person_matches_ebike(region["bbox"], ebike_bbox):
                related_boxes.append(region["bbox"])

        return related_boxes

    def _merge_bboxes(self, boxes: List[List[float]]) -> List[float]:
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    def _limit_ebike_bbox_growth(
        self,
        anchor_bbox: List[float],
        expanded_bbox: List[float],
    ) -> List[float]:
        ax1, ay1, ax2, ay2 = anchor_bbox
        anchor_w = max(1.0, ax2 - ax1)
        anchor_h = max(1.0, ay2 - ay1)
        return [
            max(expanded_bbox[0], ax1 - anchor_w * 1.00),
            max(expanded_bbox[1], ay1 - anchor_h * 0.35),
            min(expanded_bbox[2], ax2 + anchor_w * 1.00),
            min(expanded_bbox[3], ay2 + anchor_h * 0.55),
        ]

    def _bike_region_support_score(
        self,
        ebike_bbox: List[float],
        region_bbox: List[float],
    ) -> float:
        score = self._iou(ebike_bbox, region_bbox)
        ex1, ey1, ex2, ey2 = ebike_bbox
        rx1, ry1, rx2, ry2 = region_bbox
        center_x = (ex1 + ex2) / 2
        center_y = (ey1 + ey2) / 2
        if rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2:
            score += 0.35
        return score

    def _person_ebike_overlap_score(
        self,
        person_bbox: List[float],
        ebike_bbox: List[float],
    ) -> float:
        if not self._person_matches_ebike(person_bbox, ebike_bbox):
            return 0.0

        px1, py1, px2, py2 = person_bbox
        foot_point_x = (px1 + px2) / 2
        foot_point_y = py2
        ex1, ey1, ex2, ey2 = ebike_bbox
        score = 0.25
        if ex1 <= foot_point_x <= ex2 and ey1 <= foot_point_y <= ey2:
            score += 0.5
        lower_half = [px1, py1 + (py2 - py1) * 0.45, px2, py2]
        score += self._iou(lower_half, ebike_bbox)
        return score

    def _build_temporal_detections(
        self,
        observed_detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> List[Detection]:
        ebike_class_id = settings.detection.ebike_class_id
        driver_class_id = settings.detection.driver_class_id
        temporal_detections: List[Detection] = []

        for class_id in (ebike_class_id, driver_class_id):
            if any(det.class_id == class_id for det in observed_detections):
                continue

            for item in self._temporal_memory.get(class_id, []):
                age = item["age"] + 1
                if age > settings.model.temporal_max_age:
                    continue
                requires_support = class_id == driver_class_id or age > 2
                if requires_support and not self._memory_has_support(class_id, item["bbox"], observed_detections, candidate_regions):
                    continue

                floor = 0.08 if class_id == ebike_class_id else 0.10
                confidence = max(item["confidence"] * (settings.model.temporal_decay ** age), floor)
                temporal_detections.append(Detection(
                    class_id=class_id,
                    confidence=min(confidence, item["confidence"]),
                    class_name=self._class_names[class_id],
                    bbox=item["bbox"],
                ))

        return temporal_detections

    def _update_temporal_memory(self, detections: List[Detection]) -> None:
        tracked_class_ids = (
            settings.detection.ebike_class_id,
            settings.detection.driver_class_id,
        )
        updated: Dict[int, List[Dict[str, Any]]] = {}

        for class_id in tracked_class_ids:
            current = [det for det in detections if det.class_id == class_id]
            previous = self._temporal_memory.get(class_id, [])
            used_previous: set[int] = set()
            items: List[Dict[str, Any]] = []

            for det in sorted(current, key=lambda item: item.confidence, reverse=True):
                best_index = self._find_best_memory_match(det, previous, used_previous)
                hits = 1
                if best_index is not None:
                    hits = previous[best_index]["hits"] + 1
                    used_previous.add(best_index)

                items.append({
                    "bbox": det.bbox,
                    "confidence": max(det.confidence, 0.08 if class_id == settings.detection.ebike_class_id else 0.10),
                    "hits": hits,
                    "age": 0,
                })

            for index, item in enumerate(previous):
                if index in used_previous:
                    continue
                next_age = item["age"] + 1
                if next_age > settings.model.temporal_max_age:
                    continue
                items.append({
                    **item,
                    "age": next_age,
                    "confidence": item["confidence"] * settings.model.temporal_decay,
                })

            items.sort(key=lambda entry: (entry["hits"], entry["confidence"]), reverse=True)
            updated[class_id] = items[:4]

        self._temporal_memory = updated

    def _find_candidate_regions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self._candidate_model.predict(
            frame,
            conf=settings.model.candidate_conf_thresh,
            iou=settings.detection.iou_thresh,
            imgsz=max(settings.model.imgsz, 960),
            verbose=False,
        )

        candidate_regions = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes, confs, classes):
                if int(cls_id) not in self._candidate_class_ids:
                    continue
                candidate_regions.append({
                    "bbox": bbox.tolist(),
                    "confidence": float(conf),
                    "class_name": self._candidate_names[int(cls_id)],
                })

        candidate_regions.sort(key=lambda item: item["confidence"], reverse=True)
        return self._dedupe_candidate_regions(candidate_regions)

    def _dedupe_candidate_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for region in regions:
            if any(
                region["class_name"] == existing["class_name"]
                and self._iou(region["bbox"], existing["bbox"]) > 0.5
                for existing in kept
            ):
                continue
            kept.append(region)
        return kept[:10]

    def _crop_region(self, frame: np.ndarray, region: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region["bbox"]
        box_w = x2 - x1
        box_h = y2 - y1
        class_name = region["class_name"]

        if class_name == "person":
            expand_left = box_w * 1.2
            expand_right = box_w * 1.2
            expand_top = box_h * 0.5
            expand_bottom = box_h * 1.5
        else:
            expand_left = box_w * 0.5
            expand_right = box_w * 0.5
            expand_top = box_h * 0.4
            expand_bottom = box_h * 0.8

        crop_x1 = max(0, int(x1 - expand_left))
        crop_y1 = max(0, int(y1 - expand_top))
        crop_x2 = min(w, int(x2 + expand_right))
        crop_y2 = min(h, int(y2 + expand_bottom))

        return frame[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1)

    def _restore_crop_detections(
        self,
        detections: List[Detection],
        offset: Tuple[int, int],
        frame_shape: Tuple[int, ...],
    ) -> List[Detection]:
        restored = []
        offset_x, offset_y = offset
        max_h, max_w = frame_shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            restored.append(Detection(
                class_id=det.class_id,
                confidence=det.confidence,
                class_name=det.class_name,
                bbox=[
                    max(0.0, min(max_w, x1 + offset_x)),
                    max(0.0, min(max_h, y1 + offset_y)),
                    max(0.0, min(max_w, x2 + offset_x)),
                    max(0.0, min(max_h, y2 + offset_y)),
                ],
            ))

        return restored

    def _build_proxy_ebike_detections(self, region: Dict[str, Any]) -> List[Detection]:
        ebike_class_id = settings.detection.ebike_class_id
        confidence = min(max(region["confidence"] * 0.35, 0.12), 0.35)
        return [Detection(
            class_id=ebike_class_id,
            confidence=confidence,
            class_name=self._class_names[ebike_class_id],
            bbox=region["bbox"],
        )]

    def _build_proxy_person_detection(
        self,
        region: Dict[str, Any],
        ebike_detection: Detection,
        class_id: int,
    ) -> Detection:
        confidence = min(
            max(max(region["confidence"] * 0.45, ebike_detection.confidence * 0.60), 0.12),
            0.30,
        )
        return Detection(
            class_id=class_id,
            confidence=confidence,
            class_name=self._class_names[class_id],
            bbox=region["bbox"],
        )

    def _boost_detection(self, detection: Detection, floor: float, ceiling: float) -> Detection:
        return Detection(
            class_id=detection.class_id,
            confidence=min(max(detection.confidence * 4.0, floor), ceiling),
            class_name=detection.class_name,
            bbox=detection.bbox,
        )

    def _filter_detections(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, ...],
    ) -> List[Detection]:
        frame_h, frame_w = frame_shape[:2]
        ebike_class_id = settings.detection.ebike_class_id
        driver_class_id = settings.detection.driver_class_id
        passenger_class_id = settings.detection.passenger_class_id
        helmet_class_id = settings.detection.helmet_class_id
        helmet_enabled = settings.detection.helmet_detection_enabled

        filtered: List[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            if box_w < 2 or box_h < 2:
                continue

            aspect_ratio = box_w / max(box_h, 1e-6)
            touches_border = x1 <= 1 or y1 <= 1 or x2 >= frame_w - 1 or y2 >= frame_h - 1

            if det.class_id == ebike_class_id:
                if box_w < 16 or box_h < 12:
                    continue
                if aspect_ratio < 0.35 or aspect_ratio > 4.5:
                    continue
                if box_h > frame_h * 0.48:
                    continue
                if touches_border and aspect_ratio < 0.45 and det.confidence < 0.20:
                    continue
            elif det.class_id in {driver_class_id, passenger_class_id}:
                if box_w < 8 or box_h < 16:
                    continue
                if aspect_ratio < 0.15 or aspect_ratio > 1.35:
                    continue
                if box_h > frame_h * 0.55:
                    continue
            elif det.class_id == helmet_class_id:
                if not helmet_enabled:
                    continue
                if box_w < 5 or box_h < 5:
                    continue
                if aspect_ratio < 0.3 or aspect_ratio > 2.5:
                    continue
                if box_w > frame_w * 0.20 or box_h > frame_h * 0.20:
                    continue

            filtered.append(det)

        return filtered

    def _has_ebike_support(
        self,
        detection: Detection,
        candidate_regions: List[Dict[str, Any]],
        observed_detections: List[Detection],
    ) -> bool:
        for det in observed_detections:
            if det.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id} and self._person_matches_ebike(det.bbox, detection.bbox):
                return True
            if det.class_id == settings.detection.helmet_class_id and self._boxes_related(detection.bbox, det.bbox):
                return True

        for region in candidate_regions:
            if region["class_name"] in {"bicycle", "motorcycle"} and self._boxes_related(detection.bbox, region["bbox"]):
                return True

        return self._memory_has_support(
            settings.detection.ebike_class_id,
            detection.bbox,
            observed_detections,
            candidate_regions,
        )

    def _has_local_ebike_support(
        self,
        detection: Detection,
        detections: List[Detection],
    ) -> bool:
        for det in detections:
            if det is detection:
                continue
            if det.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id}:
                if self._person_matches_ebike(det.bbox, detection.bbox):
                    return True
            if det.class_id == settings.detection.helmet_class_id and self._boxes_related(detection.bbox, det.bbox):
                return True
        return False

    def _has_driver_support(
        self,
        detection: Detection,
        candidate_regions: List[Dict[str, Any]],
        observed_detections: List[Detection],
    ) -> bool:
        for det in observed_detections:
            if det.class_id == settings.detection.ebike_class_id and self._person_matches_ebike(detection.bbox, det.bbox):
                return True

        for region in candidate_regions:
            if region["class_name"] in {"bicycle", "motorcycle"} and self._person_matches_ebike(detection.bbox, region["bbox"]):
                return True

        return self._memory_has_support(
            settings.detection.driver_class_id,
            detection.bbox,
            observed_detections,
            candidate_regions,
        )

    def _has_local_driver_support(
        self,
        detection: Detection,
        detections: List[Detection],
    ) -> bool:
        for det in detections:
            if det is detection:
                continue
            if det.class_id == settings.detection.ebike_class_id and self._person_matches_ebike(detection.bbox, det.bbox):
                return True
        return False

    def _memory_has_support(
        self,
        class_id: int,
        bbox: List[float],
        observed_detections: List[Detection],
        candidate_regions: List[Dict[str, Any]],
    ) -> bool:
        if class_id == settings.detection.ebike_class_id:
            for det in observed_detections:
                if det.class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id} and self._person_matches_ebike(det.bbox, bbox):
                    return True
                if det.class_id == settings.detection.helmet_class_id and self._boxes_related(bbox, det.bbox):
                    return True

            for region in candidate_regions:
                if region["class_name"] in {"bicycle", "motorcycle"} and self._boxes_related(bbox, region["bbox"]):
                    return True
        else:
            for det in observed_detections:
                if det.class_id == settings.detection.ebike_class_id and self._person_matches_ebike(bbox, det.bbox):
                    return True

            for region in candidate_regions:
                if region["class_name"] in {"bicycle", "motorcycle"} and self._person_matches_ebike(bbox, region["bbox"]):
                    return True

        return False

    def _find_related_person_region(
        self,
        bbox: List[float],
        regions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        related = self._find_related_person_regions(bbox, regions)
        if not related:
            return None
        return related[0]

    def _find_related_person_regions(
        self,
        bbox: List[float],
        regions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        related = [region for region in regions if self._person_matches_ebike(region["bbox"], bbox)]
        related.sort(
            key=lambda item: (
                self._person_ebike_overlap_score(item["bbox"], bbox),
                item["confidence"],
            ),
            reverse=True,
        )
        return related

    def _find_unmatched_person_region(
        self,
        regions: List[Dict[str, Any]],
        riders: List[Detection],
    ) -> Optional[Dict[str, Any]]:
        for region in regions:
            if self._region_matches_any_rider(region["bbox"], riders):
                continue
            return region
        return None

    def _region_matches_any_rider(
        self,
        region_bbox: List[float],
        riders: List[Detection],
    ) -> bool:
        for rider in riders:
            overlap = self._overlap_ratio(region_bbox, rider.bbox)
            iou = self._iou(region_bbox, rider.bbox)
            center_distance = self._center_distance(region_bbox, rider.bbox)
            min_dim = max(
                1.0,
                min(
                    region_bbox[2] - region_bbox[0],
                    region_bbox[3] - region_bbox[1],
                    rider.bbox[2] - rider.bbox[0],
                    rider.bbox[3] - rider.bbox[1],
                ),
            )
            if overlap > 0.55 or iou > 0.30 or center_distance < min_dim * 0.45:
                return True
        return False


    def _person_matches_ebike(
        self,
        person_bbox: List[float],
        ebike_bbox: List[float],
    ) -> bool:
        px1, py1, px2, py2 = person_bbox
        ex1, ey1, ex2, ey2 = ebike_bbox
        point_x = (px1 + px2) / 2
        point_y = py2

        margin_x = (ex2 - ex1) * 0.12
        margin_top = (ey2 - ey1) * 0.20
        margin_bottom = (ey2 - ey1) * 0.12
        if (
            ex1 - margin_x <= point_x <= ex2 + margin_x
            and ey1 - margin_top <= point_y <= ey2 + margin_bottom
        ):
            return True

        lower_half = [px1, py1 + (py2 - py1) * 0.45, px2, py2]
        return self._iou(lower_half, ebike_bbox) > 0.03

    def _find_related_helmet(
        self,
        bbox: List[float],
        helmets: List[Detection],
    ) -> Optional[Detection]:
        related = [helmet for helmet in helmets if self._boxes_related(bbox, helmet.bbox)]
        if not related:
            return None
        related.sort(key=lambda item: item.confidence, reverse=True)
        return related[0]

    def _estimate_driver_bbox_from_helmet(
        self,
        helmet_bbox: List[float],
        frame_shape: Tuple[int, ...],
    ) -> List[float]:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = helmet_bbox
        box_w = x2 - x1
        box_h = y2 - y1
        center_x = (x1 + x2) / 2

        driver_w = box_w * 3.6
        driver_h = box_h * 5.8
        driver_x1 = max(0.0, center_x - driver_w / 2)
        driver_x2 = min(float(frame_w), center_x + driver_w / 2)
        driver_y1 = max(0.0, y1 - box_h * 0.6)
        driver_y2 = min(float(frame_h), driver_y1 + driver_h)
        return [driver_x1, driver_y1, driver_x2, driver_y2]

    def _find_best_memory_match(
        self,
        detection: Detection,
        previous: List[Dict[str, Any]],
        used_previous: set[int],
    ) -> Optional[int]:
        best_index: Optional[int] = None
        best_iou = 0.0
        for index, item in enumerate(previous):
            if index in used_previous:
                continue
            iou = self._iou(detection.bbox, item["bbox"])
            if iou > 0.2 and iou > best_iou:
                best_iou = iou
                best_index = index
        return best_index

    def _merge_detections(
        self,
        base: List[Detection],
        refined: List[Detection],
        iou_thresh: float,
    ) -> List[Detection]:
        merged = sorted(base + refined, key=lambda det: det.confidence, reverse=True)
        kept: List[Detection] = []

        for det in merged:
            if any(
                det.class_id == existing.class_id and self._iou(det.bbox, existing.bbox) > iou_thresh
                for existing in kept
            ):
                continue
            kept.append(det)

        return kept

    def _build_offsets(self, length: int, tile_length: int, stride: int) -> List[int]:
        if tile_length >= length:
            return [0]

        offsets = list(range(0, max(1, length - tile_length + 1), stride))
        last_offset = max(0, length - tile_length)
        if offsets[-1] != last_offset:
            offsets.append(last_offset)
        return offsets

    def _boxes_related(
        self,
        anchor_bbox: List[float],
        other_bbox: List[float],
        expand_ratio: float = 0.35,
    ) -> bool:
        expanded = self._expand_bbox(anchor_bbox, expand_ratio)
        if self._iou(expanded, other_bbox) > 0.01:
            return True

        other_center_x = (other_bbox[0] + other_bbox[2]) / 2
        other_center_y = (other_bbox[1] + other_bbox[3]) / 2
        return (
            expanded[0] <= other_center_x <= expanded[2]
            and expanded[1] <= other_center_y <= expanded[3]
        )

    def _expand_bbox(self, bbox: List[float], ratio: float) -> List[float]:
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        return [
            x1 - box_w * ratio,
            y1 - box_h * ratio,
            x2 + box_w * ratio,
            y2 + box_h * ratio,
        ]

    def _box_contains(
        self,
        outer_bbox: List[float],
        inner_bbox: List[float],
        margin: float = 0.0,
    ) -> bool:
        ox1, oy1, ox2, oy2 = outer_bbox
        ix1, iy1, ix2, iy2 = inner_bbox
        outer_w = ox2 - ox1
        outer_h = oy2 - oy1
        return (
            ox1 - outer_w * margin <= ix1
            and oy1 - outer_h * margin <= iy1
            and ox2 + outer_w * margin >= ix2
            and oy2 + outer_h * margin >= iy2
        )

    def _resolve_candidate_class_ids(self, class_names: Dict[int, str]) -> set[int]:
        allowed = {"person", "bicycle", "motorcycle"}
        return {class_id for class_id, name in class_names.items() if name in allowed}

    def _iou(self, box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def _overlap_ratio(self, box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        min_area = min(area_a, area_b)
        return inter_area / min_area if min_area > 0 else 0.0

    def _center_distance(self, box_a: List[float], box_b: List[float]) -> float:
        center_a_x = (box_a[0] + box_a[2]) / 2
        center_a_y = (box_a[1] + box_a[3]) / 2
        center_b_x = (box_b[0] + box_b[2]) / 2
        center_b_y = (box_b[1] + box_b[3]) / 2
        return ((center_a_x - center_b_x) ** 2 + (center_a_y - center_b_y) ** 2) ** 0.5

    def warmup(self, imgsz: tuple = (640, 640)) -> None:
        """Warmup model with dummy input."""
        if self._model:
            dummy = np.zeros((imgsz[1], imgsz[0], 3), dtype=np.uint8)
            self._model.predict(dummy, imgsz=settings.model.imgsz, verbose=False)
            if self._candidate_model is not None:
                self._candidate_model.predict(dummy, imgsz=max(settings.model.imgsz, 960), verbose=False)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        return self._class_names
