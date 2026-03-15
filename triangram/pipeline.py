import os
import cv2
import numpy as np

from .state import TriangramState
from .base import BaseInitializer, BaseRenderer, BaseEvaluator, BaseOptimizer, BaseRecorder


class TriangramPipeline:
    def __init__(self, target_image_path: str, max_width: int = 400):
        img = cv2.imread(target_image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {target_image_path}")

        # 処理を軽くするためリサイズ
        h, w = img.shape[:2]
        self.original_size = (w, h)  # cv2.resize 用 (width, height)
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        self.state = TriangramState(img)

        self.initializer: BaseInitializer = None
        self.renderer: BaseRenderer = None
        self.evaluator: BaseEvaluator = None
        self.recorder: BaseRecorder = None
        self.optimizers: list[tuple[BaseOptimizer, int]] = []

    def setup(
        self,
        init: BaseInitializer,
        renderer: BaseRenderer,
        eval: BaseEvaluator,
        recorder: BaseRecorder = None,
    ):
        self.initializer = init
        self.renderer = renderer
        self.evaluator = eval
        self.recorder = recorder

    def add_optimizer(self, optimizer: BaseOptimizer, iterations: int):
        self.optimizers.append((optimizer, iterations))

    def run(self, num_points: int, output_dir: str = "output_images"):
        if self.initializer is None or self.renderer is None or self.evaluator is None:
            raise RuntimeError("setup() must be called before run()")

        os.makedirs(output_dir, exist_ok=True)

        print("1. Initialization...")
        self.state.points = self.initializer.initialize(self.state.target_image, num_points)

        print("2. Initial Rendering...")
        self.state.current_render = self.renderer.render(self.state)
        initial_loss = self.evaluator.evaluate(self.state.target_image, self.state.current_render)
        print(f"   Initial Loss: {initial_loss:.5f}")
        cv2.imwrite(os.path.join(output_dir, "00_initial.png"), self.state.current_render)

        if self.recorder is not None:
            self.recorder.on_step(self.state.current_render)

        print("3. Starting Optimization Pipeline...")
        for idx, (optimizer, iters) in enumerate(self.optimizers):
            print(f"--- Phase {idx+1}: {optimizer.__class__.__name__} ({iters} iters) ---")

            on_step = self.recorder.on_step if self.recorder is not None else None
            optimizer.optimize(self.state, self.renderer, self.evaluator, iters, on_step=on_step)

            current_loss = self.evaluator.evaluate(self.state.target_image, self.state.current_render)
            print(f"   Phase {idx+1} Completed. Loss: {current_loss:.2f}")
            cv2.imwrite(os.path.join(output_dir, f"{idx+1:02d}_phase_completed.png"), self.state.current_render)

        print("4. Saving result...")
        proc_w = self.state.target_image.shape[1]
        result_scale = self.original_size[0] / proc_w
        result = self.renderer.render(self.state, scale=result_scale, supersample=2)
        cv2.imwrite(os.path.join(output_dir, "result.png"), result)

        if self.recorder is not None:
            self.recorder.save(output_dir)

        print("Pipeline Finished! Check the output directory.")
