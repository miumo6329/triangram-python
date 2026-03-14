import numpy as np


class TriangramState:
    def __init__(self, target_image: np.ndarray):
        self.target_image = target_image
        self.points = np.array([])  # 頂点の座標リスト (N, 2)
        self.current_render = np.zeros_like(target_image)
