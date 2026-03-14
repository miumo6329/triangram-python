import cv2
import numpy as np
from scipy.spatial import Delaunay

from .base import BaseRenderer
from .state import TriangramState


class DelaunayRenderer(BaseRenderer):
    def render(self, state: TriangramState, scale: float = 1.0, supersample: int = 1) -> np.ndarray:
        h, w = state.target_image.shape[:2]
        out_size = (int(w * scale), int(h * scale))
        ss = scale * supersample
        ss_h, ss_w = int(h * ss), int(w * ss)
        canvas = np.zeros((ss_h, ss_w, state.target_image.shape[2]), dtype=state.target_image.dtype)

        if len(state.points) < 3:
            return canvas

        # ドロネー分割（元座標で実施）
        tri = Delaunay(state.points)
        simplices = tri.simplices

        for simplex in simplices:
            src_pts = state.points[simplex]  # 元スケールの頂点
            dst_pts = src_pts * ss           # 描画スケールの頂点

            # 元画像から平均色を取得
            pt1, pt2, pt3 = src_pts
            sx_min = max(0, int(min(pt1[0], pt2[0], pt3[0])))
            sx_max = min(w, int(max(pt1[0], pt2[0], pt3[0])) + 1)
            sy_min = max(0, int(min(pt1[1], pt2[1], pt3[1])))
            sy_max = min(h, int(max(pt1[1], pt2[1], pt3[1])) + 1)

            if sx_max <= sx_min or sy_max <= sy_min:
                continue

            src_cnt = src_pts - np.array([sx_min, sy_min])
            mask = np.zeros((sy_max - sy_min, sx_max - sx_min), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(src_cnt).astype(np.int32)], 255)
            img_roi = state.target_image[sy_min:sy_max, sx_min:sx_max]
            mean_color = cv2.mean(img_roi, mask=mask)[:3]
            color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

            # キャンバスに描画
            cv2.fillPoly(canvas, [np.round(dst_pts).astype(np.int32)], color, lineType=cv2.LINE_8)

        if supersample > 1:
            return cv2.resize(canvas, out_size, interpolation=cv2.INTER_AREA)

        return canvas
