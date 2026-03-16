import math
import random
import cv2
import numpy as np
from scipy.spatial import Delaunay

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


class AdaptiveRefiner(BaseOptimizer):
    """
    誤差駆動の適応的点追加・削除。

    Split: 誤差加重値 (MSE × 三角形面積) が最大の三角形の重心に点を追加する。
    Merge: 周辺三角形の誤差合計が最小の点を削除する。

    各イテレーションで split_count 回の追加 → merge_count 回の削除を行う。
    将来的には幾何学的基準 (近接点統合・疎領域への追加) と組み合わせた
    ハイブリッド戦略に拡張できる設計とする。
    """

    def __init__(
        self,
        split_count: int = 5,
        merge_count: int = 0,
        min_points: int = 10,
        max_points: int = 2000,
    ):
        self.split_count = split_count
        self.merge_count = merge_count
        self.min_points = min_points
        self.max_points = max_points

    def _compute_triangle_stats(self, state: TriangramState):
        """三角形ごとの誤差加重値・重心・点→三角形インデックスマップを返す。

        誤差加重値 = 三角形内ピクセルの平均二乗誤差 × 三角形面積
        (大きい三角形ほど・誤差が大きいほど高スコア)
        """
        tri = Delaunay(state.points)
        simplices = tri.simplices  # (M, 3)
        h, w = state.target_image.shape[:2]

        # チャンネル方向の平均 MSE マップ (H, W)
        diff_sq = np.mean(
            (state.target_image.astype(np.float32) - state.current_render.astype(np.float32)) ** 2,
            axis=2,
        )

        n_tri = len(simplices)
        weighted_errors = np.zeros(n_tri)
        centroids = np.zeros((n_tri, 2))

        for i, simplex in enumerate(simplices):
            pts = state.points[simplex]  # (3, 2): x, y
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]

            area = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2.0
            centroids[i] = [(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0]

            if area < 1:
                continue

            bx_min = max(0, int(min(x0, x1, x2)))
            bx_max = min(w, int(max(x0, x1, x2)) + 1)
            by_min = max(0, int(min(y0, y1, y2)))
            by_max = min(h, int(max(y0, y1, y2)) + 1)

            if bx_max <= bx_min or by_max <= by_min:
                continue

            cnt = pts - np.array([bx_min, by_min])
            mask = np.zeros((by_max - by_min, bx_max - bx_min), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(cnt).astype(np.int32)], 255)

            roi = diff_sq[by_min:by_max, bx_min:bx_max]
            pixels = mask > 0
            if pixels.any():
                weighted_errors[i] = roi[pixels].mean() * area

        # 点 → 所属三角形インデックスのマップ (Merge で使用)
        point_to_tris = [[] for _ in range(len(state.points))]
        for i, simplex in enumerate(simplices):
            for pt_idx in simplex:
                point_to_tris[pt_idx].append(i)

        return weighted_errors, centroids, point_to_tris

    def _do_split(self, state: TriangramState, renderer: BaseRenderer) -> None:
        """誤差最大の三角形の重心に1点追加する。"""
        weighted_errors, centroids, _ = self._compute_triangle_stats(state)
        best = int(np.argmax(weighted_errors))
        state.points = np.vstack([state.points, centroids[best]])
        state.current_render = renderer.render(state)

    def _do_merge(self, state: TriangramState, renderer: BaseRenderer) -> None:
        """周辺誤差合計が最小の点を1つ削除する。四隅 (idx 0–3) は除外。"""
        weighted_errors, _, point_to_tris = self._compute_triangle_stats(state)
        movable = np.arange(4, len(state.points))
        neighborhood_errors = np.array([
            sum(weighted_errors[t] for t in point_to_tris[i])
            for i in movable
        ])
        candidate_idx = movable[int(np.argmin(neighborhood_errors))]
        state.points = np.delete(state.points, candidate_idx, axis=0)
        state.current_render = renderer.render(state)

    def optimize(
        self,
        state: TriangramState,
        renderer: BaseRenderer,
        evaluator: BaseEvaluator,
        iterations: int,
        on_step: callable = None,
    ):
        added_total = 0
        removed_total = 0

        for i in range(iterations):
            added_this = 0
            removed_this = 0

            # Split: 誤差最大の三角形に点を追加
            for _ in range(self.split_count):
                if len(state.points) >= self.max_points:
                    break
                self._do_split(state, renderer)
                added_this += 1
                if on_step:
                    on_step(state.current_render)

            # Merge: 周辺誤差最小の点を削除
            for _ in range(self.merge_count):
                if len(state.points) - 4 <= self.min_points:
                    break
                self._do_merge(state, renderer)
                removed_this += 1
                if on_step:
                    on_step(state.current_render)

            added_total += added_this
            removed_total += removed_this
            current_loss = evaluator.evaluate(state.target_image, state.current_render)
            print(f"      Iter {i+1}/{iterations} | Points: {len(state.points)} (+{added_this}/-{removed_this}) | Loss: {current_loss:.5f}")

        print(f"   -> Total: +{added_total} split, -{removed_total} merged. Final points: {len(state.points)}")


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
