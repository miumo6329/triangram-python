import math
import random
import numpy as np

from .base import BaseOptimizer, BaseRenderer, BaseEvaluator
from .state import TriangramState


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    焼きなまし法: 確率的に悪化を許容することで局所最適を脱出する。
    温度Tは指数的に冷却され、終盤はヒルクライムに近づく。
    """
    def __init__(self, step: int = 25, initial_temp: float = 500.0, final_temp: float = 1.0):
        self.step = step
        self.initial_temp = initial_temp
        self.final_temp = final_temp

    def optimize(self, state: TriangramState, renderer: BaseRenderer, evaluator: BaseEvaluator, iterations: int, on_step: callable = None):
        h, w = state.target_image.shape[:2]
        current_loss = evaluator.evaluate(state.target_image, state.current_render)

        # 指数冷却: T(i) = T0 * (Tf/T0)^(i/N)
        cooling_rate = (self.final_temp / self.initial_temp) ** (1.0 / iterations)
        temp = self.initial_temp

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
                print(f"      Step {i+1}/{iterations} | Loss: {current_loss:.2f} | T: {temp:.2f}")

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
