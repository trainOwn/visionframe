import math
from typing import Callable

import cv2
import numpy as np


Rect = tuple[int, int, int, int]


def draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = int(math.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    i = 0
    while i < length:
        s = i
        e = min(i + dash_len, length)
        sx, sy = int(x1 + dx * s), int(y1 + dy * s)
        ex, ey = int(x1 + dx * e), int(y1 + dy * e)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        i += dash_len + gap_len


def draw_dotted_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 2,
    step: int = 10,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = int(math.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    for i in range(0, length + 1, step):
        cx = int(x1 + dx * i)
        cy = int(y1 + dy * i)
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_rounded_rect(
    img: np.ndarray,
    rect: Rect,
    color: tuple[int, int, int],
    thickness: int = 2,
    radius: int = 12,
) -> None:
    x1, y1, x2, y2 = rect
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_gradient_fill(img: np.ndarray, rect: Rect, c1: tuple[int, int, int], c2: tuple[int, int, int], vertical: bool) -> None:
    x1, y1, x2, y2 = rect
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    if vertical:
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        grad = np.tile(ramp, (1, w))
    else:
        ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        grad = np.tile(ramp, (h, 1))
    c1v = np.array(c1, dtype=np.float32)
    c2v = np.array(c2, dtype=np.float32)
    patch = ((1.0 - grad[..., None]) * c1v + grad[..., None] * c2v).astype(np.uint8)
    img[y1 : y2 + 1, x1 : x2 + 1] = patch


def blend_fill(img: np.ndarray, rect: Rect, color: tuple[int, int, int], alpha: float) -> None:
    x1, y1, x2, y2 = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def _auto_text_scale(
    text: str,
    max_w: int,
    max_h: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    thickness: int = 1,
    min_scale: float = 0.35,
    max_scale: float = 1.1,
) -> float:
    if max_w <= 0 or max_h <= 0:
        return min_scale
    (base_w, base_h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    if base_w <= 0 or base_h <= 0:
        return min_scale
    fit = min(max_w / float(base_w), max_h / float(base_h))
    return max(min_scale, min(max_scale, fit))


def _draw_label_box(
    img: np.ndarray,
    rect: Rect,
    text: str | None,
    text_position: str,
    text_color: tuple[int, int, int],
) -> None:
    if text is None:
        return
    label = text.strip()
    if not label:
        return

    valid_positions = {"top-left", "top-right", "bottom-left", "bottom-right"}
    if text_position not in valid_positions:
        raise ValueError(f"text_position must be one of: {', '.join(sorted(valid_positions))}")

    x1, y1, x2, y2 = rect
    rw = max(1, x2 - x1 + 1)
    rh = max(1, y2 - y1 + 1)

    # Keep the label proportional to the detection box size.
    max_text_w = max(20, int(rw * 0.72))
    max_text_h = max(12, int(rh * 0.22))
    scale = _auto_text_scale(label, max_text_w, max_text_h)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

    pad_x = max(4, int(rw * 0.03))
    pad_y = max(3, int(rh * 0.03))
    box_w = tw + 2 * pad_x
    box_h = th + baseline + 2 * pad_y
    inset = max(2, min(8, int(min(rw, rh) * 0.07)))

    gap = max(2, min(10, int(min(rw, rh) * 0.08)))
    if text_position == "top-left":
        bx1 = x1 + inset
        by1 = y1 - box_h - gap
    elif text_position == "top-right":
        bx1 = x2 - inset - box_w + 1
        by1 = y1 - box_h - gap
    elif text_position == "bottom-left":
        bx1 = x1 + inset
        by1 = y2 + gap
    else:  # bottom-right
        bx1 = x2 - inset - box_w + 1
        by1 = y2 + gap

    # Keep the label outside if possible, but flip when image bounds would clip it.
    if by1 < 0 and text_position.startswith("top"):
        by1 = y2 + gap
    if by1 + box_h > img.shape[0] and text_position.startswith("bottom"):
        by1 = y1 - box_h - gap

    # Clamp the label box inside the image.
    img_h, img_w = img.shape[:2]
    bx1 = max(0, min(bx1, img_w - box_w))
    by1 = max(0, min(by1, img_h - box_h))
    bx2 = bx1 + box_w - 1
    by2 = by1 + box_h - 1

    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, dst=img)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (230, 230, 230), 1, cv2.LINE_AA)

    tx = bx1 + pad_x
    ty = by1 + pad_y + th
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


def style_solid(img: np.ndarray, rect: Rect) -> None:
    cv2.rectangle(img, rect[:2], rect[2:], (25, 220, 25), 2, cv2.LINE_AA)


def style_thick(img: np.ndarray, rect: Rect) -> None:
    cv2.rectangle(img, rect[:2], rect[2:], (20, 120, 255), 6, cv2.LINE_AA)


def style_thin_gray(img: np.ndarray, rect: Rect) -> None:
    cv2.rectangle(img, rect[:2], rect[2:], (160, 160, 160), 1, cv2.LINE_8)


def style_dashed(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    color = (0, 215, 255)
    draw_dashed_line(img, (x1, y1), (x2, y1), color, 2)
    draw_dashed_line(img, (x2, y1), (x2, y2), color, 2)
    draw_dashed_line(img, (x2, y2), (x1, y2), color, 2)
    draw_dashed_line(img, (x1, y2), (x1, y1), color, 2)


def style_dotted(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    color = (255, 255, 0)
    draw_dotted_line(img, (x1, y1), (x2, y1), color, 2, 9)
    draw_dotted_line(img, (x2, y1), (x2, y2), color, 2, 9)
    draw_dotted_line(img, (x2, y2), (x1, y2), color, 2, 9)
    draw_dotted_line(img, (x1, y2), (x1, y1), color, 2, 9)


def style_dash_dot(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    color = (255, 140, 0)
    draw_dashed_line(img, (x1, y1), (x2, y1), color, 2, 12, 6)
    draw_dotted_line(img, (x2, y1), (x2, y2), color, 2, 10)
    draw_dashed_line(img, (x2, y2), (x1, y2), color, 2, 12, 6)
    draw_dotted_line(img, (x1, y2), (x1, y1), color, 2, 10)


def style_double_border(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 3, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + 7, y1 + 7), (x2 - 7, y2 - 7), (0, 100, 255), 2, cv2.LINE_AA)


def style_rounded(img: np.ndarray, rect: Rect) -> None:
    draw_rounded_rect(img, rect, (255, 90, 90), 2, 14)


def style_rounded_thick(img: np.ndarray, rect: Rect) -> None:
    draw_rounded_rect(img, rect, (90, 220, 255), 5, 18)


def style_shadow(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1 + 5, y1 + 5), (x2 + 5, y2 + 5), (40, 40, 40), -1)
    blend_fill(img, rect, (20, 20, 20), 0.35)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)


def style_glow(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for t, c in [(12, (255, 70, 70)), (8, (255, 110, 110)), (4, (255, 160, 160))]:
        cv2.rectangle(img, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)


def style_transparent_fill(img: np.ndarray, rect: Rect) -> None:
    blend_fill(img, rect, (0, 200, 120), 0.25)
    cv2.rectangle(img, rect[:2], rect[2:], (0, 220, 130), 2, cv2.LINE_AA)


def style_gradient_horizontal(img: np.ndarray, rect: Rect) -> None:
    draw_gradient_fill(img, rect, (255, 0, 0), (0, 255, 255), vertical=False)
    cv2.rectangle(img, rect[:2], rect[2:], (255, 255, 255), 2, cv2.LINE_AA)


def style_gradient_vertical(img: np.ndarray, rect: Rect) -> None:
    draw_gradient_fill(img, rect, (200, 40, 255), (50, 220, 50), vertical=True)
    cv2.rectangle(img, rect[:2], rect[2:], (20, 20, 20), 2, cv2.LINE_AA)


def style_h_stripes(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (120, 80, 30), 0.2)
    for y in range(y1, y2 + 1, 8):
        cv2.line(img, (x1, y), (x2, y), (60, 180, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, rect[:2], rect[2:], (255, 255, 255), 2, cv2.LINE_AA)


def style_v_stripes(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (80, 40, 100), 0.2)
    for x in range(x1, x2 + 1, 8):
        cv2.line(img, (x, y1), (x, y2), (255, 210, 70), 2, cv2.LINE_AA)
    cv2.rectangle(img, rect[:2], rect[2:], (245, 245, 245), 2, cv2.LINE_AA)


def style_cross_hatch(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (50, 50, 50), 0.2)
    for k in range(-80, (x2 - x1) + 80, 12):
        cv2.line(img, (x1 + k, y1), (x1 + k + 80, y2), (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(img, (x1 + k, y2), (x1 + k + 80, y1), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(img, rect[:2], rect[2:], (255, 255, 255), 2, cv2.LINE_AA)


def style_corner_brackets(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (40, 255, 40)
    l = 20
    t = 3
    cv2.line(img, (x1, y1), (x1 + l, y1), c, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + l), c, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - l, y1), c, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + l), c, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + l, y2), c, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - l), c, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - l, y2), c, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - l), c, t, cv2.LINE_AA)


def style_cut_corner(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    d = 14
    pts = np.array(
        [[x1 + d, y1], [x2 - d, y1], [x2, y1 + d], [x2, y2 - d], [x2 - d, y2], [x1 + d, y2], [x1, y2 - d], [x1, y1 + d]],
        np.int32,
    )
    cv2.polylines(img, [pts], True, (255, 180, 0), 3, cv2.LINE_AA)


def style_neon(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (255, 60, 170), 0.12)
    for t in [10, 6, 3]:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 80, 190), t, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)


def style_sketch(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    rng = np.random.default_rng(7)
    for _ in range(4):
        j = rng.integers(-2, 3, size=8)
        pts = np.array(
            [
                [x1 + j[0], y1 + j[1]],
                [x2 + j[2], y1 + j[3]],
                [x2 + j[4], y2 + j[5]],
                [x1 + j[6], y2 + j[7]],
            ],
            np.int32,
        )
        cv2.polylines(img, [pts], True, (20, 20, 20), 1, cv2.LINE_AA)


def style_multi_color_edges(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.line(img, (x1, y1), (x2, y1), (255, 80, 80), 3, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), (80, 255, 80), 3, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x1, y2), (80, 180, 255), 3, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y1), (255, 255, 80), 3, cv2.LINE_AA)


def style_inner_border(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (80, 120, 240), 0.18)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (20, 20, 20), 2, cv2.LINE_AA)


def style_glass(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (230, 230, 230), 0.2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (250, 250, 250), 2, cv2.LINE_AA)
    cv2.line(img, (x1 + 8, y1 + 8), (x2 - 8, y1 + 8), (255, 255, 255), 1, cv2.LINE_AA)


def style_pixel(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (255, 255, 255)
    for x in range(x1, x2 + 1, 6):
        cv2.rectangle(img, (x, y1), (x + 2, y1 + 2), c, -1)
        cv2.rectangle(img, (x, y2 - 2), (x + 2, y2), c, -1)
    for y in range(y1, y2 + 1, 6):
        cv2.rectangle(img, (x1, y), (x1 + 2, y + 2), c, -1)
        cv2.rectangle(img, (x2 - 2, y), (x2, y + 2), c, -1)


def style_handdrawn_highlight(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (0, 255, 255), 0.12)
    for off in [0, 1]:
        cv2.rectangle(img, (x1 + off, y1 + off), (x2 - off, y2 - off), (30, 30, 30), 1, cv2.LINE_AA)


def style_top_glow(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for i, a in enumerate([0.35, 0.2, 0.1]):
        yb = min(y2, y1 + 8 + i * 10)
        blend_fill(img, (x1, y1, x2, yb), (255, 230, 120), a)
    cv2.rectangle(img, (x1, y1), (x2, y2), (240, 240, 240), 2, cv2.LINE_AA)


def style_bottom_glow(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for i, a in enumerate([0.35, 0.2, 0.1]):
        yt = max(y1, y2 - 8 - i * 10)
        blend_fill(img, (x1, yt, x2, y2), (120, 210, 255), a)
    cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)


def style_rainbow_steps(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    colors = [(255, 0, 80), (255, 160, 0), (255, 255, 0), (0, 255, 120), (0, 180, 255), (180, 80, 255)]
    for i, c in enumerate(colors):
        cv2.rectangle(img, (x1 + i * 2, y1 + i * 2), (x2 - i * 2, y2 - i * 2), c, 2, cv2.LINE_AA)


def style_chamfer_double(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for inset, col in [(0, (255, 220, 60)), (8, (255, 120, 30))]:
        d = 12
        pts = np.array(
            [
                [x1 + inset + d, y1 + inset],
                [x2 - inset - d, y1 + inset],
                [x2 - inset, y1 + inset + d],
                [x2 - inset, y2 - inset - d],
                [x2 - inset - d, y2 - inset],
                [x1 + inset + d, y2 - inset],
                [x1 + inset, y2 - inset - d],
                [x1 + inset, y1 + inset + d],
            ],
            np.int32,
        )
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)


def style_corner_circles(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 1, cv2.LINE_AA)
    for p in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(img, p, 7, (255, 90, 90), 2, cv2.LINE_AA)
        cv2.circle(img, p, 2, (255, 255, 255), -1, cv2.LINE_AA)


def style_scallop_edge(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    step = 12
    r = 4
    for x in range(x1, x2 + 1, step):
        cv2.circle(img, (x, y1), r, (200, 255, 120), 1, cv2.LINE_AA)
        cv2.circle(img, (x, y2), r, (200, 255, 120), 1, cv2.LINE_AA)
    for y in range(y1, y2 + 1, step):
        cv2.circle(img, (x1, y), r, (200, 255, 120), 1, cv2.LINE_AA)
        cv2.circle(img, (x2, y), r, (200, 255, 120), 1, cv2.LINE_AA)


def style_zigzag(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    amp = 5
    step = 12
    top = [(x, y1 + (amp if i % 2 else 0)) for i, x in enumerate(range(x1, x2 + 1, step))]
    bot = [(x, y2 - (amp if i % 2 else 0)) for i, x in enumerate(range(x1, x2 + 1, step))]
    left = [(x1 + (amp if i % 2 else 0), y) for i, y in enumerate(range(y1, y2 + 1, step))]
    right = [(x2 - (amp if i % 2 else 0), y) for i, y in enumerate(range(y1, y2 + 1, step))]
    cv2.polylines(img, [np.array(top, np.int32)], False, (255, 180, 100), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array(bot, np.int32)], False, (255, 180, 100), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array(left, np.int32)], False, (255, 180, 100), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array(right, np.int32)], False, (255, 180, 100), 2, cv2.LINE_AA)


def style_wave_border(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    xs = np.arange(x1, x2 + 1, 4)
    top = np.array([(int(x), int(y1 + 4 * math.sin((x - x1) * 0.2))) for x in xs], np.int32)
    bot = np.array([(int(x), int(y2 + 4 * math.sin((x - x1) * 0.2))) for x in xs], np.int32)
    ys = np.arange(y1, y2 + 1, 4)
    left = np.array([(int(x1 + 4 * math.sin((y - y1) * 0.2)), int(y)) for y in ys], np.int32)
    right = np.array([(int(x2 + 4 * math.sin((y - y1) * 0.2)), int(y)) for y in ys], np.int32)
    for pts in [top, bot, left, right]:
        cv2.polylines(img, [pts], False, (120, 255, 220), 2, cv2.LINE_AA)


def style_checker_fill(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    s = 10
    for y in range(y1, y2 + 1, s):
        for x in range(x1, x2 + 1, s):
            if ((x - x1) // s + (y - y1) // s) % 2 == 0:
                cv2.rectangle(img, (x, y), (min(x + s - 1, x2), min(y + s - 1, y2)), (170, 170, 170), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (250, 250, 250), 2, cv2.LINE_AA)


def style_noise_fill(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    rng = np.random.default_rng(11)
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    noise = rng.integers(70, 190, size=(h, w, 1), dtype=np.uint8)
    tint = np.array([40, 120, 220], dtype=np.uint8).reshape(1, 1, 3)
    patch = ((noise.astype(np.uint16) + tint.astype(np.uint16)) // 2).astype(np.uint8)
    img[y1 : y2 + 1, x1 : x2 + 1] = patch
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)


def style_diag_scanlines(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (60, 60, 60), 0.2)
    for k in range(-120, (x2 - x1) + 120, 10):
        cv2.line(img, (x1 + k, y2), (x1 + k + 80, y1), (255, 120, 120), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 220, 220), 2, cv2.LINE_AA)


def style_target_box(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (90, 255, 90), 2, cv2.LINE_AA)
    cv2.line(img, (cx - 20, cy), (cx + 20, cy), (90, 255, 90), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 20), (cx, cy + 20), (90, 255, 90), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, (90, 255, 90), 1, cv2.LINE_AA)


def style_double_brackets(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for inset, col in [(0, (255, 255, 255)), (8, (120, 220, 255))]:
        l = 18
        cv2.line(img, (x1 + inset, y1 + inset), (x1 + inset + l, y1 + inset), col, 2, cv2.LINE_AA)
        cv2.line(img, (x1 + inset, y1 + inset), (x1 + inset, y1 + inset + l), col, 2, cv2.LINE_AA)
        cv2.line(img, (x2 - inset, y1 + inset), (x2 - inset - l, y1 + inset), col, 2, cv2.LINE_AA)
        cv2.line(img, (x2 - inset, y1 + inset), (x2 - inset, y1 + inset + l), col, 2, cv2.LINE_AA)
        cv2.line(img, (x1 + inset, y2 - inset), (x1 + inset + l, y2 - inset), col, 2, cv2.LINE_AA)
        cv2.line(img, (x1 + inset, y2 - inset), (x1 + inset, y2 - inset - l), col, 2, cv2.LINE_AA)
        cv2.line(img, (x2 - inset, y2 - inset), (x2 - inset - l, y2 - inset), col, 2, cv2.LINE_AA)
        cv2.line(img, (x2 - inset, y2 - inset), (x2 - inset, y2 - inset - l), col, 2, cv2.LINE_AA)


def style_corner_dots_only(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for p in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(img, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 9, (0, 150, 255), 2, cv2.LINE_AA)


def style_top_bottom_accent(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.line(img, (x1, y1), (x2, y1), (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x2, y2), (80, 80, 80), 4, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y2), (170, 170, 170), 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), (170, 170, 170), 2, cv2.LINE_AA)


def style_engrave(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (130, 130, 130), 0.18)
    cv2.line(img, (x1, y1), (x2, y1), (40, 40, 40), 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y2), (40, 40, 40), 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)


def style_emboss(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (200, 200, 200), 0.15)
    cv2.line(img, (x1, y1), (x2, y1), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y2), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x2, y2), (70, 70, 70), 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), (70, 70, 70), 2, cv2.LINE_AA)


def style_grid_fill(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (40, 100, 170), 0.15)
    for x in range(x1, x2 + 1, 14):
        cv2.line(img, (x, y1), (x, y2), (160, 210, 255), 1, cv2.LINE_AA)
    for y in range(y1, y2 + 1, 14):
        cv2.line(img, (x1, y), (x2, y), (160, 210, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (230, 245, 255), 2, cv2.LINE_AA)


def style_notched_frame(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    n = 10
    pts = np.array(
        [
            [x1 + n, y1],
            [x2 - n, y1],
            [x2, y1 + n],
            [x2, y2 - n],
            [x2 - n, y2],
            [x1 + n, y2],
            [x1, y2 - n],
            [x1, y1 + n],
        ],
        np.int32,
    )
    cv2.polylines(img, [pts], True, (120, 255, 140), 2, cv2.LINE_AA)
    cv2.polylines(img, [pts + np.array([[-4, -4]], np.int32)], True, (60, 180, 90), 1, cv2.LINE_AA)


def style_fade_border(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for i, a in enumerate([0.35, 0.22, 0.12, 0.06]):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 - i * 2, y1 - i * 2), (x2 + i * 2, y2 + i * 2), (255, 200, 60), 2)
        cv2.addWeighted(overlay, a, img, 1 - a, 0, dst=img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)


def style_futuristic_hud(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (255, 230, 60)
    cv2.rectangle(img, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
    for l in [18, 30]:
        cv2.line(img, (x1, y1), (x1 + l, y1), c, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + l), c, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - l, y1), c, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + l), c, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + l, y2), c, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - l), c, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - l, y2), c, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - l), c, 2, cv2.LINE_AA)


def style_futuristic_laser_grid(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (90, 20, 20), 0.2)
    for x in range(x1, x2 + 1, 12):
        cv2.line(img, (x, y1), (x, y2), (255, 90, 90), 1, cv2.LINE_AA)
    for y in range(y1, y2 + 1, 12):
        cv2.line(img, (x1, y), (x2, y), (255, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 170, 170), 2, cv2.LINE_AA)


def style_futuristic_circuit(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 255, 180), 2, cv2.LINE_AA)
    xs = [x1 + 14, x1 + 28, x2 - 28, x2 - 14]
    ys = [y1 + 14, y1 + 28, y2 - 28, y2 - 14]
    for x in xs:
        cv2.line(img, (x, y1), (x, y1 + 10), (80, 255, 180), 1, cv2.LINE_AA)
        cv2.line(img, (x, y2), (x, y2 - 10), (80, 255, 180), 1, cv2.LINE_AA)
    for y in ys:
        cv2.line(img, (x1, y), (x1 + 10, y), (80, 255, 180), 1, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2 - 10, y), (80, 255, 180), 1, cv2.LINE_AA)
    for p in [(x1 + 14, y1 + 14), (x2 - 14, y1 + 14), (x1 + 14, y2 - 14), (x2 - 14, y2 - 14)]:
        cv2.circle(img, p, 2, (80, 255, 180), -1, cv2.LINE_AA)


def style_futuristic_hex_corners(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (255, 200, 80)
    d = 10
    corners = [
        np.array([[x1, y1 + d], [x1 + d, y1], [x1 + 2 * d, y1 + d], [x1 + d, y1 + 2 * d]], np.int32),
        np.array([[x2, y1 + d], [x2 - d, y1], [x2 - 2 * d, y1 + d], [x2 - d, y1 + 2 * d]], np.int32),
        np.array([[x1, y2 - d], [x1 + d, y2], [x1 + 2 * d, y2 - d], [x1 + d, y2 - 2 * d]], np.int32),
        np.array([[x2, y2 - d], [x2 - d, y2], [x2 - 2 * d, y2 - d], [x2 - d, y2 - 2 * d]], np.int32),
    ]
    for poly in corners:
        cv2.polylines(img, [poly], True, c, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), c, 1, cv2.LINE_AA)


def style_futuristic_pulse_ring(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for r, a in [(12, 0.22), (24, 0.14), (36, 0.08)]:
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, (120, 255, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, a, img, 1 - a, 0, dst=img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (120, 255, 255), 2, cv2.LINE_AA)


def style_futuristic_data_ticks(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (255, 255, 120)
    cv2.rectangle(img, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
    for x in range(x1, x2 + 1, 16):
        cv2.line(img, (x, y1), (x, y1 - 5), c, 1, cv2.LINE_AA)
        cv2.line(img, (x, y2), (x, y2 + 5), c, 1, cv2.LINE_AA)
    for y in range(y1, y2 + 1, 16):
        cv2.line(img, (x1, y), (x1 - 5, y), c, 1, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2 + 5, y), c, 1, cv2.LINE_AA)


def style_futuristic_dual_glow(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for t, c in [(8, (255, 110, 60)), (4, (70, 220, 255))]:
        cv2.rectangle(img, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)


def style_futuristic_scan_corner(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (100, 255, 120)
    l = 25
    cv2.line(img, (x1, y1), (x1 + l, y1), c, 3, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + l), c, 3, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - l, y2), c, 3, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - l), c, 3, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - l, y1), c, 1, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + l, y2), c, 1, cv2.LINE_AA)


def style_futuristic_tech_brace(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (255, 150, 90)
    m = (x1 + x2) // 2
    cv2.polylines(img, [np.array([[x1, y1], [m - 8, y1], [m, y1 + 8], [m + 8, y1], [x2, y1]], np.int32)], False, c, 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array([[x1, y2], [m - 8, y2], [m, y2 - 8], [m + 8, y2], [x2, y2]], np.int32)], False, c, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y2), c, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), c, 2, cv2.LINE_AA)


def style_futuristic_quantum(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (40, 20, 100), 0.2)
    draw_gradient_fill(img, rect, (80, 0, 120), (0, 180, 255), vertical=False)
    for i in range(0, 4):
        cv2.rectangle(img, (x1 + i * 3, y1 + i * 3), (x2 - i * 3, y2 - i * 3), (255, 255, 255), 1, cv2.LINE_AA)


def style_funny_bouncy(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    pts = []
    step = 10
    for i, x in enumerate(range(x1, x2 + 1, step)):
        pts.append((x, y1 + (4 if i % 2 else 0)))
    cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 255, 255), 3, cv2.LINE_AA)
    pts = []
    for i, x in enumerate(range(x1, x2 + 1, step)):
        pts.append((x, y2 - (4 if i % 2 else 0)))
    cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y2), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)


def style_funny_googly(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
    eye1 = (x1 + 18, y1 - 8)
    eye2 = (x2 - 18, y1 - 8)
    for e, p in [(eye1, (x1 + 20, y1 - 8)), (eye2, (x2 - 20, y1 - 8))]:
        cv2.circle(img, e, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 3, (20, 20, 20), -1, cv2.LINE_AA)


def style_funny_confetti(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    rng = np.random.default_rng(42)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 180, 0), 2, cv2.LINE_AA)
    for _ in range(60):
        x = int(rng.integers(x1, x2 + 1))
        y = int(rng.integers(y1, y2 + 1))
        c = (int(rng.integers(60, 255)), int(rng.integers(60, 255)), int(rng.integers(60, 255)))
        cv2.circle(img, (x, y), int(rng.integers(1, 3)), c, -1, cv2.LINE_AA)


def style_funny_cloud(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for x in range(x1 + 8, x2, 18):
        cv2.circle(img, (x, y1), 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y2), 7, (255, 255, 255), 2, cv2.LINE_AA)
    for y in range(y1 + 8, y2, 18):
        cv2.circle(img, (x1, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (x2, y), 7, (255, 255, 255), 2, cv2.LINE_AA)


def style_funny_wobble(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    for k in [0, 2]:
        pts = np.array(
            [
                [x1 + k, y1 + 1],
                [x2 - 3, y1 + 3 + k],
                [x2 + 1, y2 - 2],
                [x1 + 2, y2 + k],
            ],
            np.int32,
        )
        cv2.polylines(img, [pts], True, (120, 250, 250), 2, cv2.LINE_AA)


def style_funny_candy(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    colors = [(255, 80, 120), (255, 180, 80)]
    for i, x in enumerate(range(x1, x2 + 1, 10)):
        c = colors[i % 2]
        cv2.line(img, (x, y1), (x, y2), c, 4, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)


def style_funny_moustache(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)
    cx = (x1 + x2) // 2
    my = y2 + 6
    left = np.array([[cx, my], [cx - 14, my + 6], [cx - 28, my + 2], [cx - 38, my + 10]], np.int32)
    right = np.array([[cx, my], [cx + 14, my + 6], [cx + 28, my + 2], [cx + 38, my + 10]], np.int32)
    cv2.polylines(img, [left], False, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.polylines(img, [right], False, (30, 30, 30), 2, cv2.LINE_AA)


def style_funny_bubble_wrap(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (230, 230, 255), 0.2)
    for y in range(y1 + 5, y2, 12):
        for x in range(x1 + 5, x2, 12):
            cv2.circle(img, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 255), 2, cv2.LINE_AA)


def style_funny_game_ui(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_8)
    cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (0, 200, 255), 2, cv2.LINE_8)
    for p in [(x1 + 8, y1 + 8), (x2 - 8, y1 + 8), (x1 + 8, y2 - 8), (x2 - 8, y2 - 8)]:
        cv2.rectangle(img, (p[0] - 2, p[1] - 2), (p[0] + 2, p[1] + 2), (255, 255, 255), -1)


def style_funny_squiggle(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    xs = np.arange(x1, x2 + 1, 4)
    top = np.array([(int(x), int(y1 + 5 * math.sin((x - x1) * 0.35))) for x in xs], np.int32)
    bot = np.array([(int(x), int(y2 + 5 * math.sin((x - x1) * 0.35 + 1.2))) for x in xs], np.int32)
    ys = np.arange(y1, y2 + 1, 4)
    left = np.array([(int(x1 + 5 * math.sin((y - y1) * 0.35)), int(y)) for y in ys], np.int32)
    right = np.array([(int(x2 + 5 * math.sin((y - y1) * 0.35 + 1.2)), int(y)) for y in ys], np.int32)
    for pts in [top, bot, left, right]:
        cv2.polylines(img, [pts], False, (255, 255, 120), 2, cv2.LINE_AA)


def style_pro_clean_blue(img: np.ndarray, rect: Rect) -> None:
    cv2.rectangle(img, rect[:2], rect[2:], (210, 120, 30), 2, cv2.LINE_AA)


def style_pro_slate(img: np.ndarray, rect: Rect) -> None:
    blend_fill(img, rect, (90, 95, 105), 0.12)
    cv2.rectangle(img, rect[:2], rect[2:], (180, 180, 185), 2, cv2.LINE_AA)


def style_pro_minimal_black(img: np.ndarray, rect: Rect) -> None:
    cv2.rectangle(img, rect[:2], rect[2:], (20, 20, 20), 1, cv2.LINE_AA)


def style_pro_gold(img: np.ndarray, rect: Rect) -> None:
    blend_fill(img, rect, (70, 95, 130), 0.08)
    cv2.rectangle(img, rect[:2], rect[2:], (80, 170, 230), 2, cv2.LINE_AA)


def style_pro_subtle_shadow(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (40, 40, 40), -1)
    blend_fill(img, rect, (255, 255, 255), 0.06)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2, cv2.LINE_AA)


def style_pro_corner_accents(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (220, 220, 220)
    l = 12
    cv2.rectangle(img, (x1, y1), (x2, y2), (130, 130, 130), 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1 + l, y1), c, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + l), c, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - l, y1), c, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + l), c, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + l, y2), c, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - l), c, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - l, y2), c, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - l), c, 2, cv2.LINE_AA)


def style_pro_report(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    blend_fill(img, rect, (255, 255, 255), 0.08)
    cv2.rectangle(img, (x1, y1), (x2, y2), (190, 190, 190), 1, cv2.LINE_AA)
    for y in range(y1 + 12, y2, 10):
        cv2.line(img, (x1 + 8, y), (x2 - 8, y), (150, 150, 150), 1, cv2.LINE_AA)


def style_pro_dashed_gray(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    c = (175, 175, 175)
    draw_dashed_line(img, (x1, y1), (x2, y1), c, 1, 8, 6)
    draw_dashed_line(img, (x2, y1), (x2, y2), c, 1, 8, 6)
    draw_dashed_line(img, (x2, y2), (x1, y2), c, 1, 8, 6)
    draw_dashed_line(img, (x1, y2), (x1, y1), c, 1, 8, 6)


def style_pro_doubleline(img: np.ndarray, rect: Rect) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (120, 120, 120), 1, cv2.LINE_AA)


def style_pro_soft_fill(img: np.ndarray, rect: Rect) -> None:
    blend_fill(img, rect, (120, 140, 170), 0.12)
    cv2.rectangle(img, rect[:2], rect[2:], (210, 210, 210), 2, cv2.LINE_AA)


def _style_cyberpro_box(
    img: np.ndarray,
    rect: Rect,
    frame: tuple[int, int, int],
    accent: tuple[int, int, int],
    glow: tuple[int, int, int],
    variant: int,
) -> None:
    x1, y1, x2, y2 = rect
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    l = max(10, min(w, h) // 5)

    blend_fill(img, rect, glow, 0.07)
    cv2.rectangle(img, (x1, y1), (x2, y2), frame, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), accent, 1, cv2.LINE_AA)

    # Corner braces.
    cv2.line(img, (x1, y1), (x1 + l, y1), accent, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + l), accent, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - l, y1), accent, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + l), accent, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + l, y2), accent, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - l), accent, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - l, y2), accent, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - l), accent, 2, cv2.LINE_AA)

    if variant % 5 == 0:
        for x in range(x1 + 12, x2 - 8, 16):
            cv2.line(img, (x, y1 + 1), (x + 6, y1 + 1), accent, 1, cv2.LINE_AA)
            cv2.line(img, (x, y2 - 1), (x + 6, y2 - 1), accent, 1, cv2.LINE_AA)
    elif variant % 5 == 1:
        for y in range(y1 + 12, y2 - 8, 14):
            cv2.line(img, (x1 + 1, y), (x1 + 1, y + 5), accent, 1, cv2.LINE_AA)
            cv2.line(img, (x2 - 1, y), (x2 - 1, y + 5), accent, 1, cv2.LINE_AA)
    elif variant % 5 == 2:
        for x in range(x1 + 10, x2 - 10, 18):
            cv2.line(img, (x, y1 + 7), (x + 6, y1 + 13), accent, 1, cv2.LINE_AA)
            cv2.line(img, (x, y2 - 7), (x + 6, y2 - 13), accent, 1, cv2.LINE_AA)
    elif variant % 5 == 3:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.line(img, (cx - 10, y1 + 2), (cx + 10, y1 + 2), accent, 1, cv2.LINE_AA)
        cv2.line(img, (cx - 10, y2 - 2), (cx + 10, y2 - 2), accent, 1, cv2.LINE_AA)
        cv2.line(img, (x1 + 2, cy - 10), (x1 + 2, cy + 10), accent, 1, cv2.LINE_AA)
        cv2.line(img, (x2 - 2, cy - 10), (x2 - 2, cy + 10), accent, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), glow, 1, cv2.LINE_AA)


def _style_cyberpro_circle(
    img: np.ndarray,
    rect: Rect,
    frame: tuple[int, int, int],
    accent: tuple[int, int, int],
    glow: tuple[int, int, int],
    variant: int,
) -> None:
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = max(8, min((x2 - x1), (y2 - y1)) // 2 - 4)

    blend_fill(img, rect, glow, 0.05)
    cv2.rectangle(img, (x1, y1), (x2, y2), frame, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, frame, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, r - 10), accent, 1, cv2.LINE_AA)

    # HUD-like arcs.
    cv2.ellipse(img, (cx, cy), (r, r), 0, 18, 72, accent, 2, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 198, 252, accent, 2, cv2.LINE_AA)

    if variant % 5 == 0:
        cv2.line(img, (cx - r, cy), (cx + r, cy), glow, 1, cv2.LINE_AA)
    elif variant % 5 == 1:
        cv2.line(img, (cx, cy - r), (cx, cy + r), glow, 1, cv2.LINE_AA)
    elif variant % 5 == 2:
        for a in [45, 135, 225, 315]:
            rad = math.radians(a)
            px = int(cx + math.cos(rad) * r)
            py = int(cy + math.sin(rad) * r)
            cv2.circle(img, (px, py), 2, accent, -1, cv2.LINE_AA)
    elif variant % 5 == 3:
        cv2.ellipse(img, (cx, cy), (max(4, r - 18), max(4, r - 18)), 0, 110, 160, glow, 1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (max(4, r - 18), max(4, r - 18)), 0, 290, 340, glow, 1, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), 3, accent, -1, cv2.LINE_AA)


def style_cyberpro_box_1(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (255, 255, 0), (255, 180, 0), (80, 40, 0), 1)


def style_cyberpro_box_2(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (255, 200, 80), (255, 255, 180), (80, 60, 20), 2)


def style_cyberpro_box_3(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (255, 160, 80), (255, 240, 120), (70, 50, 20), 3)


def style_cyberpro_box_4(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (180, 255, 120), (240, 255, 200), (40, 70, 35), 4)


def style_cyberpro_box_5(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (120, 255, 255), (200, 255, 255), (35, 70, 70), 5)


def style_cyberpro_box_6(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (255, 255, 255), (255, 220, 200), (55, 40, 30), 6)


def style_cyberpro_box_7(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (170, 230, 255), (255, 255, 255), (50, 60, 80), 7)


def style_cyberpro_box_8(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (255, 140, 120), (255, 210, 180), (65, 45, 35), 8)


def style_cyberpro_box_9(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (220, 220, 220), (255, 255, 255), (65, 65, 65), 9)


def style_cyberpro_box_10(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_box(img, rect, (120, 255, 170), (220, 255, 240), (35, 70, 50), 10)


def style_cyberpro_circle_1(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (255, 255, 0), (255, 200, 40), (90, 40, 0), 1)


def style_cyberpro_circle_2(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (255, 220, 120), (255, 255, 220), (85, 65, 30), 2)


def style_cyberpro_circle_3(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (200, 255, 150), (255, 255, 210), (40, 80, 40), 3)


def style_cyberpro_circle_4(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (150, 255, 255), (230, 255, 255), (40, 70, 70), 4)


def style_cyberpro_circle_5(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (180, 230, 255), (255, 255, 255), (55, 70, 90), 5)


def style_cyberpro_circle_6(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (255, 170, 120), (255, 220, 180), (70, 45, 35), 6)


def style_cyberpro_circle_7(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (255, 255, 255), (220, 220, 220), (70, 70, 70), 7)


def style_cyberpro_circle_8(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (140, 255, 180), (225, 255, 245), (35, 70, 55), 8)


def style_cyberpro_circle_9(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (255, 245, 150), (255, 255, 230), (85, 75, 35), 9)


def style_cyberpro_circle_10(img: np.ndarray, rect: Rect) -> None:
    _style_cyberpro_circle(img, rect, (210, 240, 255), (255, 255, 255), (60, 70, 90), 10)


STYLES: list[tuple[str, Callable[[np.ndarray, Rect], None]]] = [
    ("solid", style_solid),
    ("thick", style_thick),
    ("thin-gray", style_thin_gray),
    ("dashed", style_dashed),
    ("dotted", style_dotted),
    ("dash-dot", style_dash_dot),
    ("double-border", style_double_border),
    ("rounded", style_rounded),
    ("rounded-thick", style_rounded_thick),
    ("shadow", style_shadow),
    ("glow", style_glow),
    ("transparent-fill", style_transparent_fill),
    ("gradient-h", style_gradient_horizontal),
    ("gradient-v", style_gradient_vertical),
    ("h-stripes", style_h_stripes),
    ("v-stripes", style_v_stripes),
    ("cross-hatch", style_cross_hatch),
    ("corner-brackets", style_corner_brackets),
    ("cut-corner", style_cut_corner),
    ("neon", style_neon),
    ("sketch", style_sketch),
    ("multi-color", style_multi_color_edges),
    ("inner-border", style_inner_border),
    ("glass", style_glass),
    ("pixel", style_pixel),
    ("highlight", style_handdrawn_highlight),
    ("top-glow", style_top_glow),
    ("bottom-glow", style_bottom_glow),
    ("rainbow-steps", style_rainbow_steps),
    ("chamfer-double", style_chamfer_double),
    ("corner-circles", style_corner_circles),
    ("scallop-edge", style_scallop_edge),
    ("zigzag", style_zigzag),
    ("wave-border", style_wave_border),
    ("checker-fill", style_checker_fill),
    ("noise-fill", style_noise_fill),
    ("diag-scanlines", style_diag_scanlines),
    ("target-box", style_target_box),
    ("double-brackets", style_double_brackets),
    ("corner-dots", style_corner_dots_only),
    ("top-bottom-accent", style_top_bottom_accent),
    ("engrave", style_engrave),
    ("emboss", style_emboss),
    ("grid-fill", style_grid_fill),
    ("notched-frame", style_notched_frame),
    ("fade-border", style_fade_border),
    ("futuristic-hud", style_futuristic_hud),
    ("futuristic-laser-grid", style_futuristic_laser_grid),
    ("futuristic-circuit", style_futuristic_circuit),
    ("futuristic-hex-corners", style_futuristic_hex_corners),
    ("futuristic-pulse-ring", style_futuristic_pulse_ring),
    ("futuristic-data-ticks", style_futuristic_data_ticks),
    ("futuristic-dual-glow", style_futuristic_dual_glow),
    ("futuristic-scan-corner", style_futuristic_scan_corner),
    ("futuristic-tech-brace", style_futuristic_tech_brace),
    ("futuristic-quantum", style_futuristic_quantum),
    ("funny-bouncy", style_funny_bouncy),
    ("funny-googly", style_funny_googly),
    ("funny-confetti", style_funny_confetti),
    ("funny-cloud", style_funny_cloud),
    ("funny-wobble", style_funny_wobble),
    ("funny-candy", style_funny_candy),
    ("funny-moustache", style_funny_moustache),
    ("funny-bubble-wrap", style_funny_bubble_wrap),
    ("funny-game-ui", style_funny_game_ui),
    ("funny-squiggle", style_funny_squiggle),
    ("pro-clean-blue", style_pro_clean_blue),
    ("pro-slate", style_pro_slate),
    ("pro-minimal-black", style_pro_minimal_black),
    ("pro-gold", style_pro_gold),
    ("pro-subtle-shadow", style_pro_subtle_shadow),
    ("pro-corner-accents", style_pro_corner_accents),
    ("pro-report", style_pro_report),
    ("pro-dashed-gray", style_pro_dashed_gray),
    ("pro-doubleline", style_pro_doubleline),
    ("pro-soft-fill", style_pro_soft_fill),
    ("cyberpro-box-1", style_cyberpro_box_1),
    ("cyberpro-box-2", style_cyberpro_box_2),
    ("cyberpro-box-3", style_cyberpro_box_3),
    ("cyberpro-box-4", style_cyberpro_box_4),
    ("cyberpro-box-5", style_cyberpro_box_5),
    ("cyberpro-box-6", style_cyberpro_box_6),
    ("cyberpro-box-7", style_cyberpro_box_7),
    ("cyberpro-box-8", style_cyberpro_box_8),
    ("cyberpro-box-9", style_cyberpro_box_9),
    ("cyberpro-box-10", style_cyberpro_box_10),
    ("cyberpro-circle-1", style_cyberpro_circle_1),
    ("cyberpro-circle-2", style_cyberpro_circle_2),
    ("cyberpro-circle-3", style_cyberpro_circle_3),
    ("cyberpro-circle-4", style_cyberpro_circle_4),
    ("cyberpro-circle-5", style_cyberpro_circle_5),
    ("cyberpro-circle-6", style_cyberpro_circle_6),
    ("cyberpro-circle-7", style_cyberpro_circle_7),
    ("cyberpro-circle-8", style_cyberpro_circle_8),
    ("cyberpro-circle-9", style_cyberpro_circle_9),
    ("cyberpro-circle-10", style_cyberpro_circle_10),
]

STYLE_MAP: dict[str, Callable[[np.ndarray, Rect], None]] = {name: fn for name, fn in STYLES}


def drawcv(
    image: np.ndarray,
    style_id: int | str,
    coords: Rect,
    color: tuple[int, int, int] | None = None,
    line_width: int = 0,
    text: str | None = None,
    text_position: str = "top-left",
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Draw a fancy rectangle style on an image and optionally overlay a custom color/line width border.

    Args:
        image: Target BGR image (modified in place).
        style_id: Style index (int) or style name (str), from STYLES.
        coords: Rectangle coordinates (x1, y1, x2, y2).
        color: Optional BGR border color for an extra overlay border.
        line_width: Extra overlay border thickness. Use 0 to disable.
        text: Optional label text to draw in a corner text box.
        text_position: Label corner: top-left, top-right, bottom-left, bottom-right.
        text_color: BGR text color for the label.
    """
    # text = "Person 0.9"
    # text_position = "top-left"

    if image is None:
        raise ValueError("image must be a valid numpy array.")
    if len(coords) != 4:
        raise ValueError("coords must be a 4-tuple: (x1, y1, x2, y2).")

    x1, y1, x2, y2 = map(int, coords)
    if x2 < x1 or y2 < y1:
        raise ValueError("coords must satisfy x2 >= x1 and y2 >= y1.")
    rect: Rect = (x1, y1, x2, y2)

    style_fn: Callable[[np.ndarray, Rect], None]
    if isinstance(style_id, int):
        if style_id < 0 or style_id >= len(STYLES):
            raise ValueError(f"style_id index out of range. Valid range: 0..{len(STYLES) - 1}")
        style_fn = STYLES[style_id][1]
    elif isinstance(style_id, str):
        if style_id not in STYLE_MAP:
            raise ValueError(f"Unknown style_id '{style_id}'. Use one of: {', '.join(STYLE_MAP.keys())}")
        style_fn = STYLE_MAP[style_id]
    else:
        raise TypeError("style_id must be int (index) or str (style name).")

    style_fn(image, rect)
    if color is not None and int(line_width) > 0:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, int(line_width), cv2.LINE_AA)
    _draw_label_box(image, rect, text, text_position, text_color)
    return image


def list_visionframe_styles() -> list[str]:
    return [name for name, _ in STYLES]


def create_visionframe_gallery(tile_w: int = 240, tile_h: int = 150, cols: int = 4) -> np.ndarray:
    rows = math.ceil(len(STYLES) / cols)
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)
    for i, (name, style_fn) in enumerate(STYLES):
        r = i // cols
        c = i % cols
        x0 = c * tile_w
        y0 = r * tile_h
        cv2.rectangle(canvas, (x0 + 8, y0 + 8), (x0 + tile_w - 8, y0 + tile_h - 8), (245, 245, 245), -1)
        cv2.rectangle(canvas, (x0 + 8, y0 + 8), (x0 + tile_w - 8, y0 + tile_h - 8), (220, 220, 220), 1)
        rect = (x0 + 30, y0 + 42, x0 + tile_w - 30, y0 + tile_h - 24)
        style_fn(canvas, rect)
        cv2.putText(canvas, name, (x0 + 14, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    gallery = create_visionframe_gallery()
    output = "visionframe_styles_gallery.png"
    cv2.imwrite(output, gallery)
    print(f"Saved: {output} ({len(STYLES)} styles)")
    cv2.imshow("visionframe Rectangle Styles", gallery)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
