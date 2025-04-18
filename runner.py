from pathlib import Path
import numpy as np
import cv2
from os import PathLike
import onnxruntime as ort
from PIL import Image
from img2texture import image_to_seamless


def file_check():
    """
    Check if the required onnx files exist in the current directory.
    """
    for file in model_names:
        if not Path(f"./data/{file}.onnx").is_file():
            print(
                f"Error: {file} not found. please download from https://github.com/armory3d/armorai/releases"
            )
            return False
    return True


def to_seamless_img(fname: str) -> np.ndarray:
    src_image = Image.open(fname)
    result_image = image_to_seamless(src_image, overlap=0.1)
    result_image = np.array(result_image)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    return result_image


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """[0,255] → [-1,1] に正規化"""
    return (img.astype(np.float32) / 255.0 - 0.5) / 0.5


def _denormalize_channel(ch: np.ndarray) -> np.ndarray:
    """[-1,1] → [0,255] に戻す"""
    return np.clip((ch * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def _run_onnx_inference(model_path: str, input_tensor: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]
    return output


def _blend_border(
    tile: np.ndarray, neighbor_tile: np.ndarray, border_w: int, vertical=False
):
    h, w, _ = tile.shape
    blended = tile.copy()
    for i in range(border_w):
        f = i / border_w
        invf = 1.0 - f
        if vertical:
            blended[i, :, :] = (
                tile[i, :, :] * f + neighbor_tile[-(i + 1), :, :] * invf
            ).astype(np.uint8)
        else:
            blended[:, i, :] = (
                tile[:, i, :] * f + neighbor_tile[:, -(i + 1), :] * invf
            ).astype(np.uint8)
    return blended


# border_w = 64
# tile_w = 1024  # 2048


def generate_texture_tiles(
    img: np.ndarray, model_type: str, tile_size=2048, border=64, single_ch=False
):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    tile_wb = tile_size + 2 * border  # tile + border
    output_img = np.zeros((h, w, 3), dtype=np.uint8)

    tiles_x = w // tile_size + 1
    tiles_y = h // tile_size + 1

    previous_tiles = {}

    for y in range(tiles_y):
        for x in range(tiles_x):
            # タイル領域 + ボーダー込みで取り出し
            sx = max(x * tile_size - border, -9999990)
            sy = max(y * tile_size - border, -9999990)
            ex = sx + tile_wb
            ey = sy + tile_wb

            patch = np.zeros((tile_wb, tile_wb, 3), dtype=np.uint8)

            w_ = w % tile_wb if x == tiles_x - 1 else tile_wb
            h_ = h % tile_wb if y == tiles_y - 1 else tile_wb

            # 画像から切り取る（端は反転などで補完）
            x_list = np.clip(list(range(sx, sx + tile_wb)), 0, w - 1)
            y_list = np.clip(list(range(sy, sy + tile_wb)), 0, h - 1)
            patch = img[np.ix_(y_list, x_list)]

            # ONNX 推論用に変換
            norm_patch = _normalize_image(patch)
            input_tensor = norm_patch.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, H, W)

            # 推論
            model_path = f"./data/photo_to_{model_type}.quant.onnx"
            output = _run_onnx_inference(model_path, input_tensor)[0]  # (3, H, W)
            output = output.transpose(1, 2, 0)  # (H, W, 3)

            # 出力画像を[-1,1]→[0,255]に変換
            rgb_result = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            if single_ch:
                # 1chの場合はRGBに変換
                res = _denormalize_channel(
                    output[border:-border, border:-border]
                )
                rgb_result = res.repeat(3, axis=2)
            else:
                for c in range(3):
                    rgb_result[:, :, c] = _denormalize_channel(
                        output[border:-border, border:-border, c]
                    )

            # Seam blending: 左・上とブレンド
            if x > 0:
                left_tile = previous_tiles[(y, x - 1)]
                rgb_result = _blend_border(
                    rgb_result, left_tile, border, vertical=False
                )

            if y > 0:
                top_tile = previous_tiles[(y - 1, x)]
                rgb_result = _blend_border(rgb_result, top_tile, border, vertical=True)

            previous_tiles[(y, x)] = output

            # 出力画像に貼り付け
            output_img[
                y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size
            ] = rgb_result[:h_, :w_]

    # RGB画像をBGRに変換
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    # cut border
    output_img = output_img[:h, :w]
    return output_img


def runner_main(fimg: PathLike, do_seamless: bool = False) -> dict:
    p = Path(fimg)
    assert p.is_file(), f"File {fimg} does not exist."

    if do_seamless:
        img = to_seamless_img(str(fimg))
    else:
        img = cv2.imread(str(fimg))

    print("img.shape:", img.shape)
    print("1 / 4 generating normal map")
    normal_img = generate_texture_tiles(img, "normal")
    print("2 / 4 generating roughness map")
    roughness_img = generate_texture_tiles(img, "roughness", single_ch=True)
    print("3 / 4 generating occlusion map")
    occlusion_img = generate_texture_tiles(img, "occlusion", single_ch=True)
    print("4 / 4 generating height map")
    height_img = generate_texture_tiles(img, "height", single_ch=True)

    return {
        "normal": normal_img,
        "roughness": roughness_img,
        "occlusion": occlusion_img,
        "height": height_img,
        "basecolor": img,
    }
