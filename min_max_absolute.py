import os

import torch
from osgeo import gdal

path_hirise_dtm = '/home/super/datasets-nas/uahirise/dtm/'
map_files = sorted(os.listdir(path_hirise_dtm))
max, min = 0, 0
for idx_f, file_name in enumerate(map_files):
    n_file_name = file_name.split('.')[0]
    print("Processing file: ", file_name, idx_f)
    dtm = gdal.Open(path_hirise_dtm+n_file_name+'.IMG')
    dtm_array=dtm.ReadAsArray()
    transformation  = dtm.GetProjection()
    dtm_tensor = torch.FloatTensor(dtm_array)
    tmp_max=torch.max(dtm_tensor)
    tmp_min=torch.min(dtm_tensor)
    if  tmp_max> max:
        max = tmp_max
    if tmp_min < min:
        min = tmp_min

print(max, min)