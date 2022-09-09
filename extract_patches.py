import numpy as np
from osgeo import gdal
from patchify import patchify
import pandas as pd
from paths import *
import random

width = 512
height = 512
step = 400


def convert_to_tiff(name):
    dtm_name = name.replace('_L', '').replace('_R', '') + '.IMG'
    associated_dtm_path = str(dtm_path / dtm_name)
    associated_dtm = gdal.Open(associated_dtm_path)

    input_filename = str(gray_path / (name + '.JP2'))
    output_filename = str(scaled_path / (name + '.tiff'))

    input_map = gdal.Open(input_filename)  # JP2 to convert
    options = gdal.WarpOptions(options=['ts'], width=associated_dtm.RasterXSize, height=associated_dtm.RasterYSize)
    converted_map = gdal.Warp(output_filename, input_map, options=options)
    return converted_map


def scale_files():
    scaled_path.mkdir(exist_ok=True)
    gray_files = [str(f.name)[:-4] for f in gray_path.iterdir() if f.is_file()]
    scaled_files = [str(f.name)[:-5] for f in scaled_path.iterdir() if f.is_file()]
    to_scale = [f for f in gray_files if f not in scaled_files]
    for i, file in enumerate(to_scale):
        converted = convert_to_tiff(file)
        print(f"{i+1}/{len(to_scale)} Cropped -> ", converted.RasterXSize, converted.RasterYSize, converted.RasterXSize,
          converted.RasterYSize)


def save_patch_gdal(path, data, proj, no_data):  # data -> 3, 512, 512
    if not np.any(data[2] == no_data):
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(str(path), data.shape[1], data.shape[2], 2, gdal.GDT_Float32)
        if random.random() >= 0.5
            outdata.GetRasterBand(1).WriteArray(data[0, :, :])  # sx
        else
            outdata.GetRasterBand(1).WriteArray(data[1, :, :])  # dx
        outdata.GetRasterBand(2).WriteArray(data[2, :, :])
        outdata.SetProjection(proj)
        outdata.FlushCache()
        print('Patch salvata')
    else:
        print('Patch scartata')


if __name__ == '__main__':
    scale_files()
    patches_path.mkdir(exist_ok=True)
    files = [str(f.name)[:-4] for f in dtm_path.iterdir() if f.is_file()]
    files = [f for f in files if not (patches_path / f).exists()]
    for i, file in enumerate(files):
        output_path = patches_path / file
        print(f'Splitting {file}')

        left_tiff = gdal.Open(str(scaled_path / (file + '_L.tiff'))).ReadAsArray()
        right_tiff = gdal.Open(str(scaled_path / (file + '_R.tiff'))).ReadAsArray()
        dtm_raster = gdal.Open(str(dtm_path / (file + '.IMG')))

        projection = dtm_raster.GetProjection()
        no_data = dtm_raster.GetRasterBand(1).GetNoDataValue()

        stack = np.stack((left_tiff, right_tiff, dtm_raster.ReadAsArray()))

        patches = patchify(stack, (3, height, width), step=step)  # 1, rows, cols, 3, height, width
        patches = patches.reshape(patches.shape[1] * patches.shape[2], patches.shape[3], patches.shape[4], -1)  # num, 3, height, width

        num = patches.shape[0]
        print(f'{num} patches generated')

        print(output_path)
        output_path.mkdir(exist_ok=True)
        for j in range(num):
            print(f"File {i + 1}/{len(files)}, Patch {j + 1}/{num}")
            patch = patches[j]
            save_patch_gdal(output_path / f'patch_{j + 1}.tiff', patch, projection, no_data)