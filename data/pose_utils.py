"""
data/pose_utils.py
==================
Filename parsing utilities for pose JSON files.

Pose JSON files produced by different extractors follow the pattern:
    <clip_name>_<extractor>_tracked_person.json

e.g.
    01_0001_alphapose_tracked_person.json
    normal_scene_01_scenario_01_alphapose_tracked_person.json
    frontdoor10_mmpose_tracked_person.json
"""
from __future__ import annotations

import re
from pathlib import Path


POSE_SUFFIX_RE = re.compile(r"_(?P<extractor>[^_]+)_tracked_person\.json$")
EXTRACTOR_MARKERS = (
    "_sam31box_",
    "_sam3box_",
    "_mmpose",
    "_alphapose",
    "_rtmpose",
    "_vitpose",
    "_yolov",
    "_ultralytics",
)


def _split_pose_filename(filename: str) -> tuple[str, str | None]:
    """Return (clip_name, extractor_name) for a tracked-pose JSON filename."""
    name = Path(filename).name
    suffix = "_tracked_person.json"
    if not name.endswith(suffix):
        return Path(name).stem, None

    stem = name[: -len(suffix)]

    m = POSE_SUFFIX_RE.search(name)
    if m:
        clip_name = name[: m.start()]
        extractor = m.group("extractor")
    else:
        clip_name, extractor = stem, None

    for marker in EXTRACTOR_MARKERS:
        idx = stem.rfind(marker)
        if idx > 0:
            return stem[:idx], stem[idx + 1:]

    return clip_name, extractor


def strip_pose_suffix(filename: str) -> str:
    """Strip '<extractor>_tracked_person.json' from a pose filename."""
    clip_name, _ = _split_pose_filename(filename)
    return clip_name
