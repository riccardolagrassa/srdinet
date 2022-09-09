import random
import shutil
import time

import numpy
import piq
import torchvision.utils
from PIL import Image
from osgeo import gdal
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from config import *
from dataset import BaseDataset_wth_folders_wthDEM


def save_tiff(d, path):
    im = Image.fromarray(d)  # float32
    im.save(path, "TIFF")


def save_tiff_gdal_singleband(path, data, transformation, min_dtm_tile, max_dtm_tile, transform_mode=False):
    data= data.squeeze(0).squeeze(0)
    data = data.cpu().numpy()
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, data.shape[0], data.shape[1], 1, gdal.GDT_Float32)
    if transform_mode:
        data = (max_dtm_tile.cpu().numpy() - min_dtm_tile.cpu().numpy()) * data - min_dtm_tile.cpu().numpy()  # TODO remove this! Do not forget!!!
        outdata.SetProjection(transformation)
    outdata.GetRasterBand(1).WriteArray(data)
    outdata.FlushCache()

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def pixelwise_precision(sr_dem_selected,dem, class_dict,final_pixelwise_abs_err_DTM_list):
    total_pixelwise_psnr_DTM_list = [[] for i in range(len(class_dict))]
    total_pixelwise_psnr_DTMTRUTH_list = [[] for i in range(len(class_dict))]
    sr_dem_flatted=sr_dem_selected.flatten().cpu().numpy()
    dem_flatted=dem.flatten().cpu().numpy()
    for idx,j in enumerate(sr_dem_flatted):
        for t1 in class_dict.keys():
            if j >= class_dict[t1][0] and j < class_dict[t1][1]:
                total_pixelwise_psnr_DTM_list[t1].append(j)
                total_pixelwise_psnr_DTMTRUTH_list[t1].append(dem_flatted[idx])
                break
    for idx1,(sublist_valuedtm, sublist_valuedtmtruth) in enumerate(zip(total_pixelwise_psnr_DTM_list,total_pixelwise_psnr_DTMTRUTH_list)):
        if sublist_valuedtmtruth and sublist_valuedtm:
            psnr_tmp = ((torch.abs(torch.tensor(sublist_valuedtmtruth) - torch.tensor(sublist_valuedtm))).mean().data)
            final_pixelwise_abs_err_DTM_list[idx1].append(psnr_tmp)


def validate(valid_dataloader, class_dict) -> float:
    total_rmseDTM_list = [[] for i in range(len(class_dict))]
    total_psnrDTM_list = [[] for i in range(len(class_dict))]
    total_psnrSR_list = [[] for i in range(len(class_dict))]
    total_s1_dtm_list = [[] for i in range(len(class_dict))]
    total_s2_dtm_list = [[] for i in range(len(class_dict))]
    total_s3_dtm_list = [[] for i in range(len(class_dict))]
    final_pixelwise_abs_err_DTM_list = [[] for i in range(len(class_dict))]


    batches = len(valid_dataloader)
    generator.eval()
    total_psnr_value, total_ssim, total_psnr_dem_value,total_ssim_dem_value, total_psnr_valueSR_Residual,total_ssimSR_Residual, total_psnr_valueDEM_Residual, total_ssimDEM_Residual = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_absolute_relative_error,total_th_metrics1,total_th_metrics2,total_th_metrics3,total_th_dtm_metrics1,total_th_dtm_metrics2,total_th_dtm_metrics3,total_rmse,total_rmse_dtm,total_absolute_relative_errorDTM = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    with torch.no_grad():
        start = time.time()
        for index, (lr, hr, dem, tile_name, transform,min_dtm_tile, max_dtm_tile) in enumerate(valid_dataloader):
            # if index == 100/batch_size:
            #     break
            if model_type == 'model_a' or model_type == 'model_b':
                if not true_super_resolution:
                    lr = lr.cuda()
                    hr = hr.cuda()
                    dem = dem.cuda()
                    sr, sr_dem_selected, rl_out = generator(lr)
                elif true_super_resolution:
                    hr = hr.cuda()
                    sr, sr_dem_selected, rl_out = generator(hr)

                #save_tiff_gdal_singleband(rlr_dir + '/' + str(index) + '_RL_OUT_', rl_out, None, None, None,transform_mode=False)


            elif model_type == 'model_c':
                if not true_super_resolution:
                    lr = lr.cuda()
                    hr = hr.cuda()
                    dem = dem.cuda()
                    sr, sr_dem_selected = generator(lr)
                elif true_super_resolution:
                    hr = hr.cuda()
                    sr, sr_dem_selected = generator(hr)

            # #torchvision.utils.save_image(sr, sr_dir + '/' + str(index) + '_SR_' + '.png')
            #save_tiff_gdal_singleband(sr_dir + '/' + str(index) + '_SR_', sr, None, None, None, transform_mode=False)
            #save_tiff_gdal_singleband(hr_dir + '/' + str(index) + '_HR_', hr, None, None, None, transform_mode=False)
            ##torchvision.utils.save_image(hr, hr_dir + '/' + str(index) + '_HR_' + '.png')
            #save_tiff_gdal_singleband(lr_dir + '/' + str(index) + '_LR_', lr, None, None, None, transform_mode=False)
            ##save_tiff_gdal_singleband(sr_extra_dir + '/' + str(index) + '_DTM_predicted', sr_dem_selected, transform[0], min_dtm_tile, max_dtm_tile, transform_mode=True)
            ##save_tiff_gdal_singleband(hr_extra_dir + '/' + str(index) + '_DTM_truth', dem, transform[0], min_dtm_tile, max_dtm_tile, transform_mode=True)

        #     if not true_super_resolution:
        #         pixelwise_precision(sr_dem_selected,dem, class_dict,final_pixelwise_abs_err_DTM_list)
        #         #Absolute error
        #         are=((torch.abs(sr - hr)).mean().data)
        #         total_absolute_relative_error += are
        #
        #         #Absolute error DTM
        #         are_dtm = ((torch.abs(sr_dem_selected - dem)).mean().data)
        #         total_absolute_relative_errorDTM += are_dtm
        #
        #         #RMSE
        #         total_rmse += RMSELoss(sr, hr)
        #         tmp_value = RMSELoss(sr_dem_selected, dem)
        #         total_rmse_dtm += tmp_value
        #         #RMSE Per class
        #         #total_rmseDTM_list[class_assigned].append(tmp_value)
        #
        #
        #         #Threshold metric SR 1.25
        #         tmp_th_s1=torch.max(sr/hr, hr/sr).double()
        #         total_th_metrics1 += torch.count_nonzero(torch.where(tmp_th_s1 < 1.25, tmp_th_s1, 0.))/(tmp_th_s1.shape[2]*tmp_th_s1.shape[3])
        #
        #         tmp_th_s2 = torch.max(sr / hr, hr / sr).double()
        #         total_th_metrics2 += torch.count_nonzero(torch.where(tmp_th_s2 < 1.25**2, tmp_th_s2, 0.)) / (tmp_th_s2.shape[2] * tmp_th_s2.shape[3])
        #
        #         tmp_th_s3 = torch.max(sr / hr, hr / sr).double()
        #         total_th_metrics3 += torch.count_nonzero(torch.where(tmp_th_s3 < 1.25**3, tmp_th_s3, 0.)) / (
        #                     tmp_th_s3.shape[2] * tmp_th_s3.shape[3])
        #
        #         # Threshold metric SR DTM 1.25
        #         tmp_th_dtms1 = torch.max(sr_dem_selected / dem, dem / sr_dem_selected).double()
        #         tmp_s_dtm_value = torch.count_nonzero(torch.where(tmp_th_dtms1 < 1.25, tmp_th_dtms1, 0.))/(tmp_th_dtms1.shape[2]*tmp_th_dtms1.shape[3])
        #         total_th_dtm_metrics1 += tmp_s_dtm_value
        #         #s1 dtm per class
        #         #total_s1_dtm_list[class_assigned].append(tmp_s_dtm_value)
        #
        #         # Threshold metric SR DTM 1.25^2
        #         tmp_th_dtms2 = torch.max(sr_dem_selected / dem, dem / sr_dem_selected).double()
        #         tmp_s_dtm_value = torch.count_nonzero(torch.where(tmp_th_dtms2 < 1.25**2, tmp_th_dtms2, 0.)) / (
        #                     tmp_th_dtms2.shape[2] * tmp_th_dtms2.shape[3])
        #         total_th_dtm_metrics2 += tmp_s_dtm_value
        #         # s2 dtm per class
        #         #total_s2_dtm_list[class_assigned].append(tmp_s_dtm_value)
        #
        #         tmp_th_dtms3 = torch.max(sr_dem_selected / dem, dem / sr_dem_selected).double()
        #         tmp_s_dtm_value=torch.count_nonzero(torch.where(tmp_th_dtms3 < 1.25**3, tmp_th_dtms3, 0.)) / (
        #                     tmp_th_dtms3.shape[2] * tmp_th_dtms3.shape[3])
        #         total_th_dtm_metrics3 += tmp_s_dtm_value
        #         # s3 dtm per class
        #         #total_s3_dtm_list[class_assigned].append(tmp_s_dtm_value)
        #
        #         #PSRN
        #         pq_psnr = piq.psnr(sr, hr, data_range=1.)
        #         pq_ssim=piq.ssim(sr, hr, data_range=1.)
        #         total_psnr_value+=pq_psnr
        #         total_ssim+= pq_ssim
        #         #PSNR per class
        #         #total_psnrSR_list[class_assigned].append(pq_psnr)
        #
        #
        #         #DTM
        #         pq_psnr_dem = piq.psnr(sr_dem_selected, dem, data_range=1.)
        #         pq_ssim_dem=piq.ssim(sr_dem_selected, dem, data_range=1.)
        #         total_psnr_dem_value+=pq_psnr_dem
        #         total_ssim_dem_value+=pq_ssim_dem
        #         #DTM per class
        #         #total_psnrDTM_list[class_assigned].append(pq_psnr_dem)
        #
        #
        # if not true_super_resolution:
        #     #SR
        #     avg_psnr_value = (total_psnr_value / (batches))
        #     avg_ssim_value = (total_ssim / (batches))
        #
        #     #DEM
        #     avg_psnr_dem_value = (total_psnr_dem_value / (batches))
        #     avg_ssim_dem_value = (total_ssim_dem_value / (batches))
        #
        #     #Absolute Relative Error
        #     avg_are_value = (total_absolute_relative_error / batches)
        #     avg_are_DTM_value = (total_absolute_relative_errorDTM / batches)
        #
        #     #threshold Error
        #     avg_th_values1 = (total_th_metrics1 / batches)/batch_size
        #     avg_th_dtm_values1 = (total_th_dtm_metrics1 / batches)/batch_size
        #     avg_th_values2 = (total_th_metrics2 / batches)/batch_size
        #     avg_th_dtm_values2 = (total_th_dtm_metrics2 / batches)/batch_size
        #     avg_th_values3 = (total_th_metrics3 / batches)/batch_size
        #     avg_th_dtm_values3 = (total_th_dtm_metrics3 / batches)/batch_size
        #
        #     #RMSE
        #     avg_rmse = (total_rmse / batches)
        #     avg_rmse_dtm = (total_rmse_dtm / batches)
        #
        #     radar_plot=False
        #     if radar_plot:
        #         new_range = 7000 + 9000  # ?
        #         new_min = 9000
        #         report_range_to_meters = []
        #         for j in class_dict.values():
        #             report_range_to_meters.append([j[0] * new_range - new_min, j[1] * new_range - new_min])
        #         import plotly.express as px
        #         import pandas as pd
        #         # abs error DTM
        #         class_dict_list = list(class_dict)
        #         df = pd.DataFrame(dict(
        #             r=[torch.sum(torch.tensor(j)).item() / len(j) if j else 0 for j in final_pixelwise_abs_err_DTM_list],
        #             theta=[str(i) for idx_i, i in enumerate(class_dict.keys())]))
        #         fig = px.line_polar(df, r='r', theta='theta', line_close=True, width=720, height=720, render_mode='auto')
        #         fig.update_layout(
        #             font=dict(
        #                 family="Courier New, monospace",
        #                 size=30,
        #                 color="RebeccaPurple"
        #             )
        #         )
        #         fig.update_traces(fill='toself')
        #         fig.write_image(working_dir + "/results/" + model_type + "_absErrorDTM_radar.png")
        #         # import plotly.express as px
        #         # import pandas as pd
        #         # #psnr DTM
        #         # df = pd.DataFrame(dict(
        #         #     r=[torch.sum(torch.tensor(j)).item()/len(j) if j else 0 for j in total_psnrDTM_list],
        #         #     theta=[str(i) for i in class_dict.keys()]))
        #         # fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        #         # fig.update_traces(fill='toself')
        #         # fig.write_image(working_dir+"/results/"+model_type+"_psnrDTM_radar.png")
        #         #
        #         # #psnr SR
        #         # df = pd.DataFrame(dict(
        #         #     r=[torch.sum(torch.tensor(j)).item() / len(j) if j else 0 for j in total_psnrSR_list],
        #         #     theta=[str(i) for i in class_dict.keys()]))
        #         # fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        #         # fig.update_traces(fill='toself')
        #         # fig.write_image(working_dir + "/results/" + model_type + "_psnrSR_radar.png")
        #         #
        #         # #s1 dtm
        #         # df = pd.DataFrame(dict(
        #         #     r=[torch.sum(torch.tensor(j)).item() / len(j) if j else 0 for j in total_s1_dtm_list],
        #         #     theta=[str(i) for i in class_dict.keys()]))
        #         # fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        #         # fig.update_traces(fill='toself')
        #         # fig.write_image(working_dir + "/results/" + model_type + "_s1_dtm_radar.png")
        #         # #s2 dtm
        #         # df = pd.DataFrame(dict(
        #         #     r=[torch.sum(torch.tensor(j)).item() / len(j) if j else 0 for j in total_s2_dtm_list],
        #         #     theta=[str(i) for i in class_dict.keys()]))
        #         # fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        #         # fig.update_traces(fill='toself')
        #         # fig.write_image(working_dir + "/results/" + model_type + "_s2_dtm_radar.png")
        #         # #s3 dtm
        #         # df = pd.DataFrame(dict(
        #         #     r=[torch.sum(torch.tensor(j)).item() / len(j) if j else 0 for j in total_s3_dtm_list],
        #         #     theta=[str(i) for i in class_dict.keys()]))
        #         # fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        #         # fig.update_traces(fill='toself')
        #         # fig.write_image(working_dir + "/results/" + model_type + "_s3_dtm_radar.png")
        #
        # if not true_super_resolution:
        #     #Metrics
        #     print("Super-Resolution evaluation")
        #     print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        #     print(f"avg RMSE SR/HR: ", avg_rmse.item())
        #     print(f"avg Absolute Error: ", avg_are_value.item())
        #     print(f"avg Threshold SR Error 1.25: ", avg_th_values1.item()," avg Threshold SR Error 1.25^2: ",avg_th_values2.item(), " avg Threshold SR Error 1.25^3: ",avg_th_values3.item())
        #     #print(f"avg per range PSNR SR: ", [(value_dict, torch.sum(torch.tensor(j)).item()/len(j)) if j else (value_dict,0) for j, value_dict in zip(total_psnrSR_list, class_dict.values())])
        #     print("#################################")
        #     print("DTM evaluation")
        #     print(f"avg PSNR DTM: {avg_psnr_dem_value:.3f} avg SSIM DEM: {avg_ssim_dem_value:.3f}")
        #     print(f"avg RMSE DTM: ", avg_rmse_dtm.item())
        #     print(f"avg Absolute DTM Error: ", avg_are_DTM_value.item())
        #     print(f"avg Threshold SR DTM Error 1.25: ", avg_th_dtm_values1.item()," avg Threshold SR DTM Error 1.25^2: ",avg_th_dtm_values2.item(), " avg Threshold SR DTM Error 1.25^3: ",avg_th_dtm_values3.item())
        #     #print(f"avg per range RMSE DTM: ", [(value_dict, torch.sum(torch.tensor(j)).item()/len(j)) if j else (value_dict,0) for j, value_dict in zip(total_rmseDTM_list, class_dict.values())])
        #     #print(f"avg per range PSNR DTM: ", [(value_dict, torch.sum(torch.tensor(j)).item()/len(j)) if j else (value_dict,0) for j, value_dict in zip(total_psnrDTM_list, class_dict.values())])
        #     print(f"avg pixel wise per range absolute error DTM: ", [(value_dict, torch.sum(torch.tensor(j)).item()/len(j)) if j else (value_dict,0) for j, value_dict in zip(final_pixelwise_abs_err_DTM_list, report_range_to_meters)])

    print("Time: ", time.time() - start)

def main() -> None:
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)
    if os.path.exists(sr_extra_dir):
        shutil.rmtree(sr_extra_dir)
    os.makedirs(sr_extra_dir)
    if os.path.exists(hr_extra_dir):
        shutil.rmtree(hr_extra_dir)
    os.makedirs(hr_extra_dir)
    if os.path.exists(hr_dir):
        shutil.rmtree(hr_dir)
    os.makedirs(hr_dir)
    if os.path.exists(lr_dir):
        shutil.rmtree(lr_dir)
    os.makedirs(lr_dir)
    if os.path.exists(sr_dir):
        shutil.rmtree(sr_dir)
    os.makedirs(sr_dir)
    if model_type == 'model_a' or model_type == 'model_b':
        if os.path.exists(rlr_dir):
            shutil.rmtree(rlr_dir)
        os.makedirs(rlr_dir)


    valid_dataset = BaseDataset_wth_folders_wthDEM(valid_dir, image_size, upscale_factor, "validate")
    my_sampler = [i for i in range(int((len(valid_dataset.filenames) * 100) / 100))]
    my_sampler=SequentialSampler(my_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True, num_workers=8, sampler=my_sampler)
    #for random test index
    #my_random_sampler = random.sample(range(len(valid_dataset.filenames)), int((len(valid_dataset.filenames) * 1) / 100))
    #my_random_sampler = SubsetRandomSampler(my_random_sampler)
    #valid_dataloader = DataLoader(valid_dataset, batch_size, False, num_workers=8, sampler=my_random_sampler)


    state_dict_Gen = torch.load(model_path_train)
    generator.load_state_dict(state_dict_Gen)
    validate(valid_dataloader, valid_dataset.a_dictionary)

if __name__ == "__main__":
    main()
