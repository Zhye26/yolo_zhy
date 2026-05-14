# YOLO Track Simple - ljt

This is a standalone project for testing YOLOv8n e-bike overload detection with different multi-object tracking methods.

It does not depend on the original Flask web app, database, training code, or original project package layout.

## Structure

```text
YOLO_TRACK_SIMPLE/
  pipeline_ljt.py              # shared YOLO/video/CSV pipeline
  gui_ljt.py                   # shared Tkinter GUI
  yolov8n_overload_gui_ljt.py  # reference script
  yolov8n_overload_video_ljt.py# reference script
  weights/
    yolov8n.pt
    osnet_x0_25_msmt17.pt
  scripts/
    run_gui_ljt.py
    run_video_association_ljt.py
    run_video_byte_ljt.py
    run_video_ocsort_ljt.py
    run_video_deepocsort_ljt.py
    run_video_botsort_ljt.py
    run_video_strongsort_ljt.py
    run_video_hybridsort_ljt.py
  methods/
    association_ljt.py
    mot_trackers_ljt/
      byte_ljt.py
      ocsort_ljt.py
      deepocsort_ljt.py
      botsort_ljt.py
      strongsort_ljt.py
      hybridsort_ljt.py
      ort_ljt.py
      oasort_ljt.py
```

## Pipeline Design

`pipeline_ljt.py` owns the shared process:

- load YOLOv8n
- read video frames
- detect `person` and `motorcycle`
- dispatch the selected MOT method
- run common person-vehicle matching
- write frame CSV, summary CSV, and optional annotated video

`gui_ljt.py` owns the GUI controls and calls the shared pipeline functions.

Concrete tracking methods live under `methods/`:

- `methods/association_ljt.py`: reference association method based on `yolov8n_overload_gui_ljt.py` and `yolov8n_overload_video_ljt.py`
- `methods/mot_trackers_ljt/`: tracker-only comparison methods backed by BoxMOT

## Current Methods

- `association`: ByteTrack tracks person and motorcycle separately; stable IDs are assigned; association binds `person stable_id` to `motorcycle stable_id`. This method no longer contains stationary hold / held-lost / recover logic.
- `iou`: simple built-in IoU tracker baseline.
- `byte`: BoxMOT ByteTrack.
- `ocsort`: BoxMOT OC-SORT.
- `deepocsort`: BoxMOT Deep OC-SORT, requires ReID weights.
- `botsort`: BoxMOT BoT-SORT, requires ReID weights.
- `strongsort`: BoxMOT StrongSORT, requires ReID weights.
- `hybridsort`: BoxMOT HybridSORT, requires ReID weights.
- `ort` / `oasort`: placeholders. The current BoxMOT installation does not provide these implementations.

See `MOT_METHOD_COMMANDS_ljt.md` for detailed Chinese notes and commands.

## Run GUI

```bash
conda activate xunienv
cd YOLO_TRACK_SIMPLE
python scripts/run_gui_ljt.py
```

In the GUI, choose the MOT backend from `tracker`.

Default weights:

```text
weights/yolov8n.pt
weights/osnet_x0_25_msmt17.pt
```

ByteTrack and OC-SORT do not require ReID weights. Deep OC-SORT, BoT-SORT, StrongSORT, and HybridSORT do.

## Run Video CLI

Association baseline:

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/video.mp4 \
  --out /path/to/output_association.mp4
```

Other launchers:

```bash
python scripts/run_video_byte_ljt.py --video /path/to/video.mp4 --out /path/to/byte.mp4
python scripts/run_video_ocsort_ljt.py --video /path/to/video.mp4 --out /path/to/ocsort.mp4
python scripts/run_video_deepocsort_ljt.py --video /path/to/video.mp4 --out /path/to/deepocsort.mp4 --reid-weights weights/osnet_x0_25_msmt17.pt
python scripts/run_video_botsort_ljt.py --video /path/to/video.mp4 --out /path/to/botsort.mp4 --reid-weights weights/osnet_x0_25_msmt17.pt
python scripts/run_video_strongsort_ljt.py --video /path/to/video.mp4 --out /path/to/strongsort.mp4 --reid-weights weights/osnet_x0_25_msmt17.pt
python scripts/run_video_hybridsort_ljt.py --video /path/to/video.mp4 --out /path/to/hybridsort.mp4 --reid-weights weights/osnet_x0_25_msmt17.pt
```

Unified entry:

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/video.mp4 \
  --out /path/to/output.mp4 \
  --tracker ocsort
```

## Output

The GUI saves results next to the selected source video.

For each selected video:

- `{video_stem}-yolov8n-overload-{tracker}-ljt_frames.csv`
- `{video_stem}-yolov8n-overload-{tracker}-ljt_summary.csv`
- `{video_stem}-yolov8n-overload-{tracker}-ljt.mp4` if `Save annotated videos` is enabled.

Important CSV fields:

- `vehicle_track_id`: motorcycle track/stable ID, such as `M001` for association.
- `matched_person_ids`: bound stable person IDs when available, such as `P003 P007`.
- `matched_person_count`: number of matched people.
- `match_scores`: person-vehicle matching scores.
- `raw_overload`: current matched person count is at least 2.
- `confirmed_overload`: overload confirmed across `confirm_frames`.

## Adding New MOT Methods

For tracker-only comparison methods, add a module under:

```text
methods/mot_trackers_ljt/<tracker_name>_ljt.py
```

Expose:

```python
def create_tracker(config):
    ...
```

Then register it in `methods/mot_trackers_ljt/factory_ljt.py`.
