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