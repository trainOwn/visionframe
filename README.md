# drawcv

`drawcv` is a computer-vision helper library that upgrades plain OpenCV bounding boxes into polished, style-rich frames for detection overlays, demos, and dashboards.

It works directly with `numpy` images and OpenCV, so you can drop it into existing object detection pipelines with minimal code change.

## Installation

```bash
pip install drawcv
```

## Why drawcv

- Replace plain `cv2.rectangle(...)` output with modern frame styles.
- Keep your current OpenCV flow (`cv2.imread`, model inference, draw, `cv2.imwrite`).
- Choose style by name or index.
- Optionally add a custom color/line overlay on top of any style.

## Quick Start

```python
import cv2
from drawcv import drawcv

image = cv2.imread("resource/test.png")

drawcv(
    image=image,
    style_id="pro-clean-blue",  # or style index like 0, 1, 2...
    coords=(80, 60, 280, 220),  # x1, y1, x2, y2
)

cv2.imwrite("output.jpg", image)
```

## OpenCV Comparison

### Standard OpenCV

```python
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

### drawcv Replacement

```python
from drawcv import drawcv

drawcv(
    image=image,
    style_id="pro-clean-blue",
    coords=(x1, y1, x2, y2),
)
```

### With Optional Overlay Color/Line Width

```python
drawcv(
    image=image,
    style_id="futuristic-hud",
    coords=(x1, y1, x2, y2),
    color=(255, 255, 255),  # BGR
    line_width=1,
)
```

## Available Styles

```python
from drawcv import list_visionframe_styles

print(list_visionframe_styles())
```

## Style Gallery

Current preview of frame styles:

![drawcv style gallery](resource/visionframe_styles_gallery.png)

Generate this gallery again:

```python
import cv2
from drawcv import create_visionframe_gallery

gallery = create_visionframe_gallery()
cv2.imwrite("resource/visionframe_styles_gallery.png", gallery)
```

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Build Package

```bash
python -m build
```