import os
import cv2
import numpy as np

from triangram import (
    TriangramPipeline,
    RandomInitializer, EdgeAwareInitializer,
    DelaunayRenderer,
    MSEEvaluator,
    SimpleRandomOptimizer,
)


if __name__ == "__main__":
    INPUT_IMAGE = r"input_images\sample.jpg"
    OUTPUT_DIR = r"output_images"

    # テスト用画像がなければ作成
    if not os.path.exists(INPUT_IMAGE):
        print("Creating a dummy test image...")
        dummy = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(dummy, (200, 150), 80, (0, 0, 255), -1)
        cv2.rectangle(dummy, (50, 50), (120, 120), (0, 255, 0), -1)
        cv2.imwrite(INPUT_IMAGE, dummy)

    # パイプラインの作成
    pipeline = TriangramPipeline(INPUT_IMAGE)

    # モジュールのセットアップ
    pipeline.setup(
        init=EdgeAwareInitializer(
            edge_ratio=0.4,
            canny_low=50,
            canny_high=250,
            bilateral_d=9,
            bilateral_sigma=75,
            debug_dir=OUTPUT_DIR
        ),
        # init=RandomInitializer(debug_dir=OUTPUT_DIR),
        renderer=DelaunayRenderer(),
        eval=MSEEvaluator(),
    )

    # 最適化フェーズの追加
    pipeline.add_optimizer(SimpleRandomOptimizer(step=50), iterations=500)
    pipeline.add_optimizer(SimpleRandomOptimizer(step=25), iterations=500)
    # pipeline.add_optimizer(SimpleRandomOptimizer(step=5), iterations=500)

    # 実行
    pipeline.run(num_points=200, output_dir=OUTPUT_DIR)
