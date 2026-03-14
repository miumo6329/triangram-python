import random
import numpy as np
import cv2

from .base import BaseInitializer


class RandomInitializer(BaseInitializer):
    def initialize(self, target_image: np.ndarray, num_points: int) -> np.ndarray:
        h, w = target_image.shape[:2]
        # 四隅は必ず配置する（端まで綺麗に描画するため）
        points = [[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]]

        # 残りの点をランダムに配置
        for _ in range(max(0, num_points - 4)):
            points.append([random.randint(0, w - 1), random.randint(0, h - 1)])

        return np.array(points, dtype=np.float32)


class EdgeAwareInitializer(BaseInitializer):
    """Cannyエッジ検出結果をもとにエッジ上に優先的に点を配置するInitializer。

    Args:
        edge_ratio: エッジ上に配置する点の割合 (0.0〜1.0)。残りはランダム配置。
        canny_low: Cannyエッジ検出の低閾値。
        canny_high: Cannyエッジ検出の高閾値。
    """

    def __init__(self, edge_ratio: float = 0.7, canny_low: int = 50, canny_high: int = 150):
        self.edge_ratio = edge_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high

    def initialize(self, target_image: np.ndarray, num_points: int) -> np.ndarray:
        h, w = target_image.shape[:2]

        # 四隅は必ず配置する（端まで綺麗に描画するため）
        corners = [[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]]
        remaining = max(0, num_points - 4)

        # Cannyエッジ検出
        gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) if target_image.ndim == 3 else target_image
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # エッジピクセルの座標を取得 (y, x) → (x, y)
        edge_ys, edge_xs = np.where(edges > 0)
        edge_points_xy = np.stack([edge_xs, edge_ys], axis=1)

        num_edge = min(int(remaining * self.edge_ratio), len(edge_points_xy))
        num_random = remaining - num_edge

        # エッジ上の点をランダムサンプリング
        if num_edge > 0 and len(edge_points_xy) > 0:
            idx = np.random.choice(len(edge_points_xy), size=num_edge, replace=len(edge_points_xy) < num_edge)
            sampled_edge = edge_points_xy[idx].tolist()
        else:
            sampled_edge = []

        # 残りをランダム配置
        sampled_random = [[random.randint(0, w - 1), random.randint(0, h - 1)] for _ in range(num_random)]

        points = corners + sampled_edge + sampled_random
        return np.array(points, dtype=np.float32)
