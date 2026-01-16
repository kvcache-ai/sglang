#!/usr/bin/env python3
"""
上下合并两张图片

用法:
    python combine_images.py <top_image> <bottom_image> -o <output_path>
"""

import argparse
from pathlib import Path

from PIL import Image


def combine_vertical(top_path: str, bottom_path: str, output_path: str, gap: int = 10):
    """上下合并两张图片"""
    top_img = Image.open(top_path)
    bottom_img = Image.open(bottom_path)

    # 如果宽度不同，调整为相同宽度
    max_width = max(top_img.width, bottom_img.width)
    if top_img.width != max_width:
        ratio = max_width / top_img.width
        top_img = top_img.resize((max_width, int(top_img.height * ratio)), Image.LANCZOS)
    if bottom_img.width != max_width:
        ratio = max_width / bottom_img.width
        bottom_img = bottom_img.resize((max_width, int(bottom_img.height * ratio)), Image.LANCZOS)

    # 创建新图片
    total_height = top_img.height + bottom_img.height + gap
    combined = Image.new('RGB', (max_width, total_height), color='white')

    # 粘贴图片
    combined.paste(top_img, (0, 0))
    combined.paste(bottom_img, (0, top_img.height + gap))

    # 保存
    combined.save(output_path, dpi=(150, 150))
    print(f"已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="上下合并两张图片")
    parser.add_argument("top", type=str, help="上方图片路径")
    parser.add_argument("bottom", type=str, help="下方图片路径")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--gap", type=int, default=10, help="两图之间的间隔像素 (默认: 10)")

    args = parser.parse_args()

    if not Path(args.top).exists():
        print(f"错误: 文件不存在 - {args.top}")
        return 1
    if not Path(args.bottom).exists():
        print(f"错误: 文件不存在 - {args.bottom}")
        return 1

    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combine_vertical(args.top, args.bottom, args.output, args.gap)
    return 0


if __name__ == "__main__":
    exit(main())
