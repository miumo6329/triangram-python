from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import BaseEvaluator


class MSEEvaluator(BaseEvaluator):
    def evaluate(self, target_image: np.ndarray, rendered_image: np.ndarray) -> float:
        # 0〜1 に正規化した上で MSE を計算（loss 範囲: 0〜1）
        diff = target_image.astype(np.float32) / 255.0 - rendered_image.astype(np.float32) / 255.0
        return float(np.mean(diff ** 2))


class SSIMEvaluator(BaseEvaluator):
    """SSIM (Structural Similarity Index) ベースの評価器。
    loss = 1 - SSIM (小さいほど良い)。
    """

    def __init__(self, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03):
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2

    def evaluate(self, target_image: np.ndarray, rendered_image: np.ndarray) -> float:
        x = target_image.astype(np.float32) / 255.0
        y = rendered_image.astype(np.float32) / 255.0

        L = 1.0
        C1 = (self.k1 * L) ** 2
        C2 = (self.k2 * L) ** 2

        def filt(img):
            # チャンネルごとにガウシアンフィルタを適用
            return np.stack([gaussian_filter(img[..., c], sigma=self.sigma) for c in range(img.shape[2])], axis=-1)

        mu_x = filt(x)
        mu_y = filt(y)
        mu_xx = filt(x * x)
        mu_yy = filt(y * y)
        mu_xy = filt(x * y)

        sigma_x2 = mu_xx - mu_x ** 2
        sigma_y2 = mu_yy - mu_y ** 2
        sigma_xy = mu_xy - mu_x * mu_y

        numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim_map = numerator / (denominator + 1e-8)
        return float(1.0 - np.mean(ssim_map))


class WeightedEvaluator(BaseEvaluator):
    """複数の Evaluator を重み付き合成する。"""

    def __init__(self, evaluators: List[Tuple[BaseEvaluator, float]]):
        """
        Args:
            evaluators: (evaluator, weight) のリスト。weight の合計が 1 になるよう正規化される。
        """
        total = sum(w for _, w in evaluators)
        self._evaluators = [(e, w / total) for e, w in evaluators]

    def evaluate(self, target_image: np.ndarray, rendered_image: np.ndarray) -> float:
        return sum(w * e.evaluate(target_image, rendered_image) for e, w in self._evaluators)
