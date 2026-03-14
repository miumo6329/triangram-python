import cv2
import numpy as np
from scipy.spatial import Delaunay

from .base import BaseRenderer
from .state import TriangramState


class DelaunayRenderer(BaseRenderer):
    def render(self, state: TriangramState) -> np.ndarray:
        h, w = state.target_image.shape[:2]
        canvas = np.zeros_like(state.target_image)

        if len(state.points) < 3:
            return canvas

        # ドロネー分割
        tri = Delaunay(state.points)
        simplices = tri.simplices
        pts = state.points

        for simplex in simplices:
            triangle_pts = pts[simplex]

            # バウンディングボックスの計算
            pt1, pt2, pt3 = triangle_pts
            x_min = max(0, int(min(pt1[0], pt2[0], pt3[0])))
            x_max = min(w, int(max(pt1[0], pt2[0], pt3[0])) + 1)
            y_min = max(0, int(min(pt1[1], pt2[1], pt3[1])))
            y_max = min(h, int(max(pt1[1], pt2[1], pt3[1])) + 1)

            if x_max <= x_min or y_max <= y_min:
                continue

            # ROIとマスクの作成
            tri_cnt = triangle_pts - np.array([x_min, y_min])
            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            cv2.fillPoly(mask, [tri_cnt.astype(np.int32)], 255)

            # 元画像の該当エリアから平均色を取得
            img_roi = state.target_image[y_min:y_max, x_min:x_max]
            mean_color = cv2.mean(img_roi, mask=mask)[:3]
            color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

            # キャンバスに描画
            cv2.fillPoly(canvas, [triangle_pts.astype(np.int32)], color, lineType=cv2.LINE_AA)

        return canvas
