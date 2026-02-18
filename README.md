# visionframe

`visionframe` is a lightweight Python package for drawing polished, style-rich bounding boxes with OpenCV.

## Installation

```bash
pip install visionframe
```

## Quick Start

```python
import cv2
from visionframe import draw_visionframe_rectangle

image = cv2.imread("input.jpg")
draw_visionframe_rectangle(
    image=image,
    style_id="pro-clean-blue",
    coords=(80, 60, 280, 220),
)
cv2.imwrite("output.jpg", image)
```

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Build Distributions

```bash
python -m build
```

Artifacts are written to `dist/`.

## Publish

TestPyPI:

```bash
twine upload --repository testpypi dist/*
```

PyPI:

```bash
twine upload dist/*
```

Set your credentials using either a `~/.pypirc` file or `TWINE_USERNAME`/`TWINE_PASSWORD`.
