"""
.trgmファイルを画像に変換するCLIツール。

使い方:
    uv run python trgm2img.py result.trgm
    uv run python trgm2img.py result.trgm output.png
    uv run python trgm2img.py result.trgm output.png --scale 2.0
"""
import argparse
import os
import sys

import cv2

from triangram import trgm


def main():
    parser = argparse.ArgumentParser(description=".trgmファイルを画像に変換する")
    parser.add_argument("input", help="入力 .trgm ファイルのパス")
    parser.add_argument("output", nargs="?", help="出力画像ファイルのパス (省略時: 入力と同名の .png)")
    parser.add_argument("--scale", type=float, default=1.0, help="出力解像度の倍率 (デフォルト: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} が見つかりません", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.splitext(args.input)[0] + ".png"

    image = trgm.render(args.input, scale=args.scale)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
