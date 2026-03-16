import json

import cv2
import numpy as np
from scipy.spatial import Delaunay

from .state import TriangramState


def _compute_triangle_colors(points: np.ndarray, target_image: np.ndarray) -> list[list[int]]:
    """各三角形の平均色(RGB)を計算する"""
    h, w = target_image.shape[:2]
    tri = Delaunay(points)
    colors = []

    for simplex in tri.simplices:
        src_pts = points[simplex]
        pt1, pt2, pt3 = src_pts
        sx_min = max(0, int(min(pt1[0], pt2[0], pt3[0])))
        sx_max = min(w, int(max(pt1[0], pt2[0], pt3[0])) + 1)
        sy_min = max(0, int(min(pt1[1], pt2[1], pt3[1])))
        sy_max = min(h, int(max(pt1[1], pt2[1], pt3[1])) + 1)

        if sx_max <= sx_min or sy_max <= sy_min:
            colors.append([0, 0, 0])
            continue

        src_cnt = src_pts - np.array([sx_min, sy_min])
        mask = np.zeros((sy_max - sy_min, sx_max - sx_min), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(src_cnt).astype(np.int32)], 255)
        img_roi = target_image[sy_min:sy_max, sx_min:sx_max]
        b, g, r = cv2.mean(img_roi, mask=mask)[:3]
        colors.append([int(r), int(g), int(b)])  # BGR → RGB

    return tri.simplices.tolist(), colors


def save(path: str, state: TriangramState) -> None:
    """.trgmファイルに保存する"""
    h, w = state.target_image.shape[:2]
    norm_vertices = (state.points / np.array([w, h])).tolist()
    triangles, colors = _compute_triangle_colors(state.points, state.target_image)

    data = {
        "version": "1.0",
        "canvas": {"width": w, "height": h},
        "vertices": norm_vertices,
        "triangles": triangles,
        "colors": colors,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def render(path: str, scale: float = 1.0) -> np.ndarray:
    """.trgmファイルを画像(BGR ndarray)としてレンダリングする"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    w = int(data["canvas"]["width"] * scale)
    h = int(data["canvas"]["height"] * scale)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    vertices = np.array(data["vertices"]) * np.array([w, h])

    for (i0, i1, i2), (r, g, b) in zip(data["triangles"], data["colors"]):
        pts = vertices[[i0, i1, i2]]
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], (b, g, r), lineType=cv2.LINE_8)

    return canvas


def load_state(path: str, target_image: np.ndarray) -> TriangramState:
    """.trgmファイルからTriangramStateを復元する"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    h, w = target_image.shape[:2]
    norm_vertices = np.array(data["vertices"], dtype=np.float64)
    points = norm_vertices * np.array([w, h])

    state = TriangramState(target_image)
    state.points = points
    return state
