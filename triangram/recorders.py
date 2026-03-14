import os
import cv2
import numpy as np
import imageio

from .base import BaseRecorder


class AnimationRecorder(BaseRecorder):
    """
    最適化ステップごとにフレームを収集し、GIF/MP4として書き出す。

    Parameters
    ----------
    interval : int
        何ステップごとにフレームを記録するか
    fps : int
        出力アニメーションのフレームレート
    formats : list[str]
        出力形式。"gif" / "mp4" の組み合わせ
    """

    def __init__(self, interval: int = 10, fps: int = 20, formats: list[str] = None):
        self.interval = interval
        self.fps = fps
        self.formats = formats if formats is not None else ["gif"]
        self._frames: list[np.ndarray] = []
        self._step_count = 0

    def on_step(self, render: np.ndarray) -> None:
        self._step_count += 1
        if self._step_count % self.interval == 0:
            self._frames.append(render.copy())

    def save(self, output_dir: str) -> None:
        if not self._frames:
            return

        print(f"5. Saving animation ({len(self._frames)} frames)...")
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self._frames]

        if "gif" in self.formats:
            path = os.path.join(output_dir, "optimization.gif")
            imageio.mimsave(path, rgb_frames, fps=self.fps, loop=0)
            print(f"   GIF saved: {path}")

        if "mp4" in self.formats:
            path = os.path.join(output_dir, "optimization.mp4")
            h, w = self._frames[0].shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for f in self._frames:
                writer.write(f)
            writer.release()
            print(f"   MP4 saved: {path}")
