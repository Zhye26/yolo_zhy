from __future__ import annotations

from typing import Any, Callable, Dict

from .base_ljt import tracker_config_from_args


AVAILABLE_TRACKERS = {
    "byte": "methods.mot_trackers_ljt.byte_ljt",
    "ocsort": "methods.mot_trackers_ljt.ocsort_ljt",
    "deepocsort": "methods.mot_trackers_ljt.deepocsort_ljt",
    "botsort": "methods.mot_trackers_ljt.botsort_ljt",
    "strongsort": "methods.mot_trackers_ljt.strongsort_ljt",
    "hybridsort": "methods.mot_trackers_ljt.hybridsort_ljt",
    "ort": "methods.mot_trackers_ljt.ort_ljt",
    "oasort": "methods.mot_trackers_ljt.oasort_ljt",
}

REID_TRACKERS = {"deepocsort", "botsort", "strongsort", "hybridsort"}


def create_mot_tracker(name: str, args: Any):
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "bytetrack": "byte",
        "byte": "byte",
        "ocsort": "ocsort",
        "deepsort": "deepocsort",
        "deepocsort": "deepocsort",
        "botsort": "botsort",
        "bot": "botsort",
        "strongsort": "strongsort",
        "strongsortplusplus": "strongsort",
        "hybridsort": "hybridsort",
        "ort": "ort",
        "oasort": "oasort",
    }
    key = aliases.get(normalized)
    if key is None or key not in AVAILABLE_TRACKERS:
        valid = ", ".join(["association", "iou"] + sorted(AVAILABLE_TRACKERS))
        raise RuntimeError(f"Unknown tracker '{name}'. Valid trackers: {valid}")

    module = __import__(AVAILABLE_TRACKERS[key], fromlist=["create_tracker"])
    config = tracker_config_from_args(args)
    return module.create_tracker(config), key
