import random
import numpy as np

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
