import numpy as np
import SimpleITK as sitk
from PIL import Image

HU_MIN, HU_MAX = -1000, 500  
PATCH_SIZE = 128              


def mha_to_image(mha_path, coord_x, coord_y, coord_z, patch_size=PATCH_SIZE):
    img = sitk.ReadImage(mha_path)
    vol = sitk.GetArrayFromImage(img)  # 形狀 (Z, Y, X)，數值是 HU

    # CSV 的 CoordX/Y/Z 是世界座標 (mm)，要換成陣列索引；SimpleITK 回傳 (x, y, z) 順序
    x, y, z = img.TransformPhysicalPointToIndex((coord_x, coord_y, coord_z))

    # 取 z-1, z, z+1 三張切片當成 R、G、B 三個通道（2.5D），剛好符合 ViT 要的 3 通道
    zs = np.clip([z - 1, z, z + 1], 0, vol.shape[0] - 1)
    slices = vol[zs]  # (3, Y, X)

    # HU 值套肺窗，壓到 0~255
    slices = np.clip(slices, HU_MIN, HU_MAX)
    slices = ((slices - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
       
    # 以結節為中心裁切；靠近邊緣時用 0（黑色）補齊
    half = patch_size // 2
    padded = np.pad(slices, ((0, 0), (half, half), (half, half)))
    patch = padded[:, y:y + patch_size, x:x + patch_size]  # pad 後原本的 (y, x) 位移了 half

    return Image.fromarray(patch.transpose(1, 2, 0))  # (Y, X, 3) → RGB 圖片


if __name__ == "__main__":
    import csv
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "mini_luna25_10.csv")) as f:
        row = next(csv.DictReader(f))

    path = os.path.join(here, "mini_images_10", row["SeriesInstanceUID"] + ".mha")
    image = mha_to_image(path, float(row["CoordX"]), float(row["CoordY"]), float(row["CoordZ"]))
    print(image.mode, image.size, "label =", row["label"])
    image.save(os.path.join(here, "example_patch.png"))
