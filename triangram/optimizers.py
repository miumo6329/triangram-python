import math
import random
import numpy as np

from .base import BaseOptimizer, BaseRenderer, BaseEvaluator
from .state import TriangramState


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    焼きなまし法: 確率的に悪化を許容することで局所最適を脱出する。
    温度Tは指数的に冷却され、終盤はヒルクライムに近づく。

    initial_temp / final_temp を省略すると自動キャリブレーション:
      - calibration_steps 回のランダム移動で mean|Δloss| を計測
      - initial_acceptance / final_acceptance の受理率になるよう温度を設定
    """
    def __init__(
        self,
        step: int = 25,
        initial_temp: float = None,
        final_temp: float = None,
        initial_acceptance: float = 0.8,
        final_acceptance: float = 0.02,
        calibration_steps: int = 50,
    ):
        self.step = step
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.initial_acceptance = initial_acceptance
        self.final_acceptance = final_acceptance
        self.calibration_steps = calibration_steps

    def _calibrate(self, state: TriangramState, renderer: BaseRenderer, evaluator: BaseEvaluator) -> tuple[float, float]:
        """mean|Δloss| を計測して (initial_temp, final_temp) を返す。"""
        h, w = state.target_image.shape[:2]
        base_loss = evaluator.evaluate(state.target_image, state.current_render)
        deltas = []
        for _ in range(self.calibration_steps):
            idx = random.randint(4, len(state.points) - 1)
            orig = state.points[idx].copy()
            dx = random.randint(-self.step, self.step)
            dy = random.randint(-self.step, self.step)
            state.points[idx] = [np.clip(orig[0] + dx, 0, w - 1), np.clip(orig[1] + dy, 0, h - 1)]
            r = renderer.render(state)
            deltas.append(abs(evaluator.evaluate(state.target_image, r) - base_loss))
            state.points[idx] = orig
        avg = max(float(np.mean(deltas)), 1e-10)
        t0 = -avg / math.log(self.initial_acceptance)
        tf = -avg / math.log(self.final_acceptance)
        return t0, tf

    def optimize(self, state: TriangramState, renderer: BaseRenderer, evaluator: BaseEvaluator, iterations: int, on_step: callable = None):
        h, w = state.target_image.shape[:2]
        current_loss = evaluator.evaluate(state.target_image, state.current_render)

        # 温度設定（省略時は自動キャリブレーション）
        if self.initial_temp is None or self.final_temp is None:
            t0, tf = self._calibrate(state, renderer, evaluator)
            initial_temp = self.initial_temp if self.initial_temp is not None else t0
            final_temp   = self.final_temp   if self.final_temp   is not None else tf
            print(f"      [auto-calibrated] T0={initial_temp:.5f}, Tf={final_temp:.5f}")
        else:
            initial_temp = self.initial_temp
            final_temp   = self.final_temp

        # 指数冷却: T(i) = T0 * (Tf/T0)^(i/N)
        cooling_rate = (final_temp / initial_temp) ** (1.0 / iterations)
        temp = initial_temp

        improved_count = 0
        accepted_worse_count = 0

        for i in range(iterations):
            idx = random.randint(4, len(state.points) - 1)
            original_pt = state.points[idx].copy()

            dx = random.randint(-self.step, self.step)
            dy = random.randint(-self.step, self.step)
            new_x = np.clip(original_pt[0] + dx, 0, w - 1)
            new_y = np.clip(original_pt[1] + dy, 0, h - 1)
            state.points[idx] = [new_x, new_y]

            new_render = renderer.render(state)
            new_loss = evaluator.evaluate(state.target_image, new_render)
            delta = new_loss - current_loss

            # 改善 or 確率的に悪化を許容
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_loss = new_loss
                state.current_render = new_render
                if delta < 0:
                    improved_count += 1
                else:
                    accepted_worse_count += 1
            else:
                state.points[idx] = original_pt

            temp *= cooling_rate

            if on_step is not None:
                on_step(state.current_render)

            if (i + 1) % 10 == 0:
                print(f"      Step {i+1}/{iterations} | Loss: {current_loss:.5f} | T: {temp:.5f}")

        print(f"   -> Improved: {improved_count}, Accepted worse: {accepted_worse_count}")


class SimpleRandomOptimizer(BaseOptimizer):
    """
    ランダムに頂点を1つ選び、少し動かしてみてLossが下がれば採用するヒルクライム法
    """
    def __init__(self, step: int = 25):
        self.step = step

    def optimize(self, state: TriangramState, renderer: BaseRenderer, evaluator: BaseEvaluator, iterations: int, on_step: callable = None):
        h, w = state.target_image.shape[:2]
        current_loss = evaluator.evaluate(state.target_image, state.current_render)

        improved_count = 0

        for i in range(iterations):
            # 四隅以外の頂点をランダムに1つ選ぶ
            idx = random.randint(4, len(state.points) - 1)
            original_pt = state.points[idx].copy()

            # ランダムに少し動かす
            dx = random.randint(-self.step, self.step)
            dy = random.randint(-self.step, self.step)
            new_x = np.clip(original_pt[0] + dx, 0, w - 1)
            new_y = np.clip(original_pt[1] + dy, 0, h - 1)
            state.points[idx] = [new_x, new_y]

            # 再描画と評価（※現在は画像全体を再描画しているため重い）
            new_render = renderer.render(state)
            new_loss = evaluator.evaluate(state.target_image, new_render)

            # 判定
            if new_loss < current_loss:
                current_loss = new_loss
                state.current_render = new_render
                improved_count += 1
            else:
                state.points[idx] = original_pt

            if on_step is not None:
                on_step(state.current_render)

            if (i + 1) % 10 == 0:
                print(f"      Step {i+1}/{iterations} | Current Loss: {current_loss:.2f}")

        print(f"   -> Optimized {improved_count} times in this phase.")
