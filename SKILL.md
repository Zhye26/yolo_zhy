---
name: ljt-naming-convention
description: Use this project-specific convention whenever creating new scripts, processing methods, experiment outputs, environments, configuration variants, folders, or generated artifacts in this repository. Newly created names should include the -ljt marker so they are easy to distinguish from upstream or existing project assets.
---

# LJT Naming Convention

When creating new project assets in this repository, add the `-ljt` marker to the name.

## Apply To

Use this convention for newly created:

- scripts
- processing methods or pipeline variants
- experiment output folders
- video/image/CSV/result artifacts
- environment files or environment names
- config variants
- model comparison outputs
- one-off validation tools

## Naming Rule

Prefer adding `-ljt` before the file extension or at the end of a directory/environment name.

Examples:

```text
validate_trained_ebike_models_video-ljt.py
trained_ebike_web_style-ljt.mp4
trained_ebike_web_style_summary-ljt.csv
model_compare_video_vis-ljt/
docker-compose.cpu-ljt.yml
ebike-validation-ljt
```

For Python identifiers where hyphens are invalid, use `_ljt`.

Examples:

```python
def validate_trained_models_ljt(...):
    ...

class WebStyleValidatorLjt:
    ...
```

## Do Not Rename Existing Assets By Default

Do not rename existing files, folders, models, dataset paths, or public API names just to add `-ljt`.

Only apply the marker to assets created from this point forward, unless the user explicitly asks to rename existing assets.

## Preserve Required External Names

If a tool or framework requires a specific filename, command, module name, or environment variable, keep the required name and add `-ljt` to the nearest safe artifact instead.

Examples:

```text
Required: docker-compose.yml
Use instead for a new variant: docker-compose-ljt.yml

Required import-safe Python module:
Use: validate_trained_ebike_models_video_ljt.py
```

## Final Response

When new assets are created, mention the `-ljt` naming in the final response so it is clear the convention was applied.
