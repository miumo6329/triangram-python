import numpy as np

from .base import BaseEvaluator


class MSEEvaluator(BaseEvaluator):
    def evaluate(self, target_image: np.ndarray, rendered_image: np.ndarray) -> float:
        # Mean Squared Error (平均二乗誤差)
        diff = target_image.astype(np.float32) - rendered_image.astype(np.float32)
        return float(np.mean(diff ** 2))
