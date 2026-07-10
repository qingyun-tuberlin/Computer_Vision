from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import random
import math


def make_image_brick_wall(
    image_dir,
    output_path="output/image_brick_wall.png",
    n_images=30,
    tile_size=100,
    gap=18,
    cols=8,
    background=(245, 245, 245),
    seed=45
):
    """
    从 image_dir 随机抽图片，生成适合放入 PPT 的砖块式图片墙。

    参数:
        image_dir: 图片文件夹路径，例如 "images"
        output_path: 输出图片路径
        n_images: 随机抽多少张图片
        tile_size: 每个小方块大小
        gap: 方块之间距离
        cols: 每行多少张
        background: 背景颜色
        seed: 随机种子，保证每次结果可复现
    """

    random.seed(seed)

    image_dir = Path(image_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    selected = random.sample(
        image_paths,
        min(n_images, len(image_paths))
    )

    rows = math.ceil(len(selected) / cols)

    canvas_w = cols * tile_size + (cols + 1) * gap
    canvas_h = rows * tile_size + (rows + 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), background)

    for idx, img_path in enumerate(selected):
        row = idx // cols
        col = idx % cols

        x = gap + col * (tile_size + gap)
        y = gap + row * (tile_size + gap)

        # 轻微随机偏移，让它不那么死板
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)

        img = Image.open(img_path).convert("RGB")

        # 中心裁剪成正方形
        img = ImageOps.fit(
            img,
            (tile_size, tile_size),
            method=Image.Resampling.LANCZOS
        )

        # 阴影，制造“凸起砖块”效果
        shadow_offset = 8
        shadow = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 90))
        shadow = shadow.filter(ImageFilter.GaussianBlur(8))

        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_layer.paste(shadow, (x + shadow_offset, y + shadow_offset))

        canvas = Image.alpha_composite(
            canvas.convert("RGBA"),
            shadow_layer
        ).convert("RGB")

        # 白色边框
        bordered = ImageOps.expand(img, border=5, fill="white")

        canvas.paste(bordered, (x, y))

    canvas.save(output_path, quality=95)
    print(f"Saved to {output_path}")


make_image_brick_wall(
    image_dir="mouse_dataset/images_perfect",
    output_path="output/image_perfect_brick_wall.png",
    n_images=4,
    tile_size=120,
    gap=16,
    cols=2
)