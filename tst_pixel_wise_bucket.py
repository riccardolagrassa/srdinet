import time
import numpy
import torch


def pixelwise_precision(sr_dem_selected,dem, class_dict):
    total_pixelwise_psnr_DTM_list = [[] for i in range(len(class_dict))]
    total_pixelwise_psnr_DTMTRUTH_list = [[] for i in range(len(class_dict))]
    sr_dem_flatted=sr_dem_selected.flatten()
    dem_flatted=dem.flatten()
    for idx,j in enumerate(sr_dem_flatted):
        for t1 in class_dict.keys():
            if class_dict[t1][0] <= j < class_dict[t1][1]:
                total_pixelwise_psnr_DTM_list[t1].append(j)
                total_pixelwise_psnr_DTMTRUTH_list[t1].append(dem_flatted[idx])
                break
    return total_pixelwise_psnr_DTM_list, total_pixelwise_psnr_DTMTRUTH_list


def pixelwise_torch(sr_dem_selected,dem, class_dict):
    l = torch.tensor([0.])
    epsilon = torch.tensor([1e-10])
    slope = torch.rand((len(class_dict)))
    for range_items in class_dict.values():
        tmp_single_lower_range_matrix=torch.tensor([range_items[0]],dtype=torch.float32)
        tmp_single_upper_range_matrix=torch.tensor([range_items[1]],dtype=torch.float32)

        lower_range_matrix = tmp_single_lower_range_matrix.repeat(64,64)
        upper_range_matrix = tmp_single_upper_range_matrix.repeat(64,64)
        the_range_mask=torch.where((sr_dem_selected >= lower_range_matrix) & (sr_dem_selected < upper_range_matrix), torch.tensor([1.]), torch.tensor([0.]))
        sr_dem_selected_masked = sr_dem_selected * the_range_mask
        dem_masked = dem * the_range_mask
        l+= -torch.log((1. / (1. + (torch.abs((sr_dem_selected_masked - dem_masked + epsilon) / (dem_masked + epsilon)) ** (slope[range_items][0]))))).mean().data
    return l/len(class_dict)



len_range = [i for i in numpy.arange(0, 1, 0.05)]
a_dictionary = dict.fromkeys(range(len(len_range) - 1))
for idx_data in range(len(len_range) - 1):
    a_dictionary[idx_data] = [len_range[idx_data], len_range[idx_data + 1]]

print(a_dictionary)
sr_dem_selected,dem = torch.rand(64,64,dtype=torch.float),torch.rand(64,64,dtype=torch.float)
start = time.time()
pixelwise_torch(sr_dem_selected, dem, a_dictionary)
# pixelwise_precision(sr_dem_selected, dem, a_dictionary)
print(time.time() - start)