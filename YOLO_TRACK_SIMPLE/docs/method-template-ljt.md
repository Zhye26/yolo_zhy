# Method Template - ljt

This project now separates the shared YOLO/video pipeline from concrete MOT methods.

## Shared Entry Points

```text
pipeline_ljt.py
```

Owns:

- YOLOv8n loading and inference
- video read/write
- detector output conversion
- selected tracker dispatch
- common person-vehicle matching
- CSV and summary output

```text
gui_ljt.py
```

Owns:

- video selection
- tracker selection
- parameter controls
- preview and progress display

## Method Locations

Association method:

```text
methods/association_ljt.py
```

Tracker-only comparison methods:

```text
methods/mot_trackers_ljt/<tracker_name>_ljt.py
```

Each tracker module should expose:

```python
def create_tracker(config):
    ...
```

Then register it in:

```text
methods/mot_trackers_ljt/factory_ljt.py
```

The GUI and CLI select it through:

```bash
--tracker <name>
```

## Recommended Common CSV Fields

Keep these fields whenever possible:

- `vehicle_track_id`
- `vehicle_class`
- `vehicle_conf`
- `vehicle_bbox`
- `matched_person_ids`
- `matched_person_count`
- `match_scores`
- `raw_overload`
- `confirmed_overload`
- `elapsed_ms`
- `fps`

Method-specific diagnostics can be added after the common fields, for example:

- `tracker_confidence`
- `reid_score`
- `association_state`

## Adding A New Launcher

Create a script under `scripts/`, for example:

```text
scripts/run_video_newtracker_ljt.py
```

Use this pattern:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_ljt import main

if __name__ == "__main__":
    if "--tracker" not in sys.argv:
        sys.argv.extend(["--tracker", "newtracker"])
    main()
```
