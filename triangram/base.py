import numpy as np
from abc import ABC, abstractmethod

from .state import TriangramState


class BaseInitializer(ABC):
    @abstractmethod
    def initialize(self, target_image: np.ndarray, num_points: int) -> np.ndarray:
        pass


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, state: TriangramState) -> np.ndarray:
        pass


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, target_image: np.ndarray, rendered_image: np.ndarray) -> float:
        pass


class BaseRecorder(ABC):
    @abstractmethod
    def on_step(self, render: np.ndarray) -> None:
        pass

    @abstractmethod
    def save(self, output_dir: str) -> None:
        pass


class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        state: TriangramState,
        renderer: BaseRenderer,
        evaluator: BaseEvaluator,
        iterations: int,
        on_step: callable = None,
    ):
        pass
