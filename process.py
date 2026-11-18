import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

def window(cube: np.ndarray, wl: float, ww: float) -> np.ndarray:
    hu_min = wl - (ww / 2.0)
    hu_max = wl + (ww / 2.0)
    clipped = np.clip(cube, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min)).astype(np.float32)

def cube_to_25planes(cube: np.ndarray) -> dict:
    c = cube.shape[0] // 2
    return {
        "axial":    cube[c],                   # (64, 64, 3)
        "coronal":  np.flipud(cube[:, c]),     # (64, 64, 3)
        "sagittal": np.flipud(cube[:, :, c]),  # (64, 64, 3)
    }

def preImg(file_path: str, real_index: tuple):
    new_spacing = (1.0, 1.0, 1.0)
    img = sitk.ReadImage(file_path)
    ori_size = img.GetSize()
    ori_spacing = img.GetSpacing()
    
    # 1. Spatial Resampling (1mm isotropic)
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(ori_size, ori_spacing, new_spacing)
    ]
    img_1mm = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        -1000,
        img.GetPixelID()
    )

    # 2. World coordinates (mm) -> Voxel index (X, Y, Z)
    voxel_idx = img_1mm.TransformPhysicalPointToIndex(real_index)

    # 3. Pad & Crop to 64x64x64
    img_to_num = sitk.GetArrayFromImage(img_1mm)
    half_width = 64 // 2

    fill_null_pixel = np.pad(
        img_to_num, 
        pad_width=half_width, 
        mode='constant', 
        constant_values=-1000
    )

    voxel_idx = np.array(voxel_idx) + half_width
    lung_nodule = fill_null_pixel[
        voxel_idx[2] - half_width : voxel_idx[2] + half_width,  
        voxel_idx[1] - half_width : voxel_idx[1] + half_width,  
        voxel_idx[0] - half_width : voxel_idx[0] + half_width   
    ]

    # 防呆檢查：確保裁切結果嚴格為 (64, 64, 64)
    assert lung_nodule.shape == (64, 64, 64), f"裁切尺寸異常: {lung_nodule.shape}"

    # 4. Multi-windowing & Compose 3D RGB Cube
    lung_cube        = window(lung_nodule, wl=-600, ww=1500)  # R: 肺窗
    mediastinal_cube = window(lung_nodule, wl=40,   ww=400)   # G: 縱膈窗
    bone_cube        = window(lung_nodule, wl=300,  ww=1500)  # B: 骨窗

    cube = np.stack([lung_cube, mediastinal_cube, bone_cube], axis=-1)  # (64, 64, 64, 3)

    # 5. Extract 2.5D Orthogonal Planes
    cube_25D = cube_to_25planes(cube)
    return cube, cube_25D


# main
if __name__ == "__main__":
    # 以這個檔案所在的資料夾為基準，從哪裡執行都找得到
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'test', 'mini_luna25_10.csv'))
    dataset_dir = os.path.join(base_dir, 'test', 'mini_images_10') + '/'
    df['file_path'] = dataset_dir + df['SeriesInstanceUID'].astype(str) + '.mha'

    pre_img = df.apply(
    lambda row: preImg(
        str(row['file_path']),
        (float(row['CoordX']), float(row['CoordY']), float(row['CoordZ']))
    ),
    axis=1
)