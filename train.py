import random
import time
import piq
from torch.utils.data import DataLoader, SubsetRandomSampler

from config import *
from dataset import BaseDataset_wth_folders_wthDEM
from validate import RMSELoss


def one_train(train_dataloader, epoch, class_dict) -> None:
    batches = len(train_dataloader)
    generator.train()
    if model_type == 'model_a' or model_type == 'model_c':
        discriminator.train()

    for index, (lr, hr, dem,tile_name,transform, min_dtm_tile, max_dtm_tile) in enumerate(train_dataloader):
        lr = lr.cuda()
        hr = hr.type(lr.dtype).cuda()
        dem = dem.type(lr.dtype).cuda()

        if model_type == 'model_a' or model_type == 'model_c':
            label_size = lr.size(0)
            real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, requires_grad=False).cuda()
            fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, requires_grad=False).cuda()

        g_optimizer.zero_grad()

        if model_type == 'model_a' or model_type == 'model_b':
            sr, sr_dem_selected, rl_out = generator(lr)

        elif model_type == 'model_c':
            sr, sr_dem_selected = generator(lr)

        loss = pixel_weight * criterion(sr, hr) + dtm_weight * criterion(sr_dem_selected, dem)


        # if epoch < warmup_epochs:
        #     print(f"Warmup active, Train Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
        #           f"Loss: {loss.item():.6f}.")
        #     loss.backward()
        #     g_optimizer.step()
        #     downSNet_optimizer.step()
        #     continue

        if model_type == 'model_a':
            hr_package = torch.cat((hr, dem), dim=1).type(lr.dtype)
            sr_package = torch.cat((sr, sr_dem_selected), dim=1).type(lr.dtype)
            hr_output = discriminator(hr_package.detach())
            sr_output = discriminator(sr_package)
            loss_GAN=adversarial_criterion(torch.abs(hr_output.detach() - sr_output), fake_label)
            g_loss= adversarial_weight * loss_GAN + loss
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()

            sr, sr_dem_selected,rl_out = generator(lr)
            sr_package = torch.cat((sr, sr_dem_selected), dim=1).type(lr.dtype)

            hr_output = discriminator(hr_package)
            sr_output = discriminator(sr_package.detach())

            d_loss_hr = adversarial_criterion(hr_output, real_label)
            d_loss_sr = adversarial_criterion(sr_output, fake_label)

            d_loss = 0.01 * ((d_loss_hr + d_loss_sr)/2)
            d_loss.backward()
            d_optimizer.step()
            if (index + 1) % 1 == 0 or (index + 1) == batches:
                print(f"Train Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                      f"Gan Loss: {loss_GAN.item():.6f} "
                      f"Loss: {loss.item():.6f} "
                      f"G Loss: {g_loss.item():.6f} "
                      f"D Loss: {d_loss.item():.6f} ")

        if model_type == 'model_c':
            hr_package = torch.cat((hr, dem), dim=1).type(lr.dtype)
            sr_package = torch.cat((sr, sr_dem_selected), dim=1).type(lr.dtype)
            hr_output = discriminator(hr_package.detach())
            sr_output = discriminator(sr_package)
            loss_GAN = adversarial_criterion(torch.abs(hr_output.detach() - sr_output), fake_label)
            g_loss = adversarial_weight * loss_GAN + loss
            g_loss.backward()
            g_optimizer.step()
            d_optimizer.zero_grad()

            sr, sr_dem_selected = generator(lr)
            sr_package = torch.cat((sr, sr_dem_selected), dim=1).type(lr.dtype)

            hr_output = discriminator(hr_package)
            sr_output = discriminator(sr_package.detach())

            d_loss_hr = adversarial_criterion(hr_output, real_label)
            d_loss_sr = adversarial_criterion(sr_output, fake_label)

            d_loss = 0.01 * ((d_loss_hr + d_loss_sr) / 2)
            d_loss.backward()
            d_optimizer.step()
            if (index + 1) % 50 == 0 or (index + 1) == batches:
                print(f"Train Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                      f"Gan Loss: {loss_GAN.item():.6f} "
                      f"Loss: {loss.item():.6f} "
                      f"G Loss: {g_loss.item():.6f} "
                      f"D Loss: {d_loss.item():.6f} ")

        if model_type == 'model_b':
            loss.backward()
            g_optimizer.step()
            if (index + 1) % 50 == 0 or (index + 1) == batches:
                print(f"Train Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                      f"Loss: {loss.item():.6f} ")
    #Scheduler steps
    scheduler.step()
    if model_type == 'model_a' or model_type == 'model_c':
        d_scheduler.step()


def validate(valid_dataloader) -> float:
    batches = len(valid_dataloader)
    generator.eval()
    total_psnr_value, total_ssim, total_psnr_dem_value,total_ssim_dem_value, total_psnr_valueSR_Residual,total_ssimSR_Residual, total_psnr_valueDEM_Residual, total_ssimDEM_Residual = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_absolute_relative_error,total_th_metrics1,total_th_metrics2,total_th_metrics3,total_th_dtm_metrics1,total_th_dtm_metrics2,total_th_dtm_metrics3,total_rmse,total_rmse_dtm,total_absolute_relative_errorDTM = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    with torch.no_grad():
        for index, (lr, hr, dem, tile_name,transform, min_dtm_tile, max_dtm_tile) in enumerate(valid_dataloader):
            lr = lr.cuda()
            hr = hr.cuda()
            dem = dem.cuda()
            if model_type == 'model_a' or model_type == 'model_b':
                sr, sr_dem_selected, rl_out = generator(lr)
            elif model_type == 'model_c':
                sr, sr_dem_selected = generator(lr)

            #Absolute error
            are=((torch.abs(sr - hr)).mean().data)
            total_absolute_relative_error += are
            #Absolute error DTM
            are_dtm = ((torch.abs(sr_dem_selected - dem)).mean().data)
            total_absolute_relative_errorDTM += are_dtm

            #RMSE
            total_rmse += RMSELoss(sr, hr)
            total_rmse_dtm += RMSELoss(sr_dem_selected, dem)


            #Threshold metric SR 1.25
            # tmp_th_s1=torch.max(sr/hr, hr/sr).double()
            # total_th_metrics1 += torch.count_nonzero(torch.where(tmp_th_s1 < 1.25, tmp_th_s1, 0.))/(tmp_th_s1.shape[2]*tmp_th_s1.shape[3])
            #
            # tmp_th_s2 = torch.max(sr / hr, hr / sr).double()
            # total_th_metrics2 += torch.count_nonzero(torch.where(tmp_th_s2 < 1.25**2, tmp_th_s2, 0.)) / (
            #             tmp_th_s2.shape[2] * tmp_th_s2.shape[3])
            #
            # tmp_th_s3 = torch.max(sr / hr, hr / sr).double()
            # total_th_metrics3 += torch.count_nonzero(torch.where(tmp_th_s3 < 1.25**3, tmp_th_s3, 0.)) / (
            #             tmp_th_s3.shape[2] * tmp_th_s3.shape[3])
            #
            # # Threshold metric SR DTM 1.25
            # tmp_th_dtms1 = torch.max(sr_dem_selected / dem, dem / sr_dem_selected).double()
            # total_th_dtm_metrics1 += torch.count_nonzero(torch.where(tmp_th_dtms1 < 1.25, tmp_th_dtms1, 0.))/(tmp_th_dtms1.shape[2]*tmp_th_dtms1.shape[3])
            #
            # tmp_th_dtms2 = torch.max(sr_dem_selected / dem, dem / sr_dem_selected).double()
            # total_th_dtm_metrics2 += torch.count_nonzero(torch.where(tmp_th_dtms2 < 1.25**2, tmp_th_dtms2, 0.)) / (
            #             tmp_th_dtms2.shape[2] * tmp_th_dtms2.shape[3])
            #
            # tmp_th_dtms3 = torch.max(sr_dem_selected / dem, dem / sr_dem_selected).double()
            # total_th_dtm_metrics3 += torch.count_nonzero(torch.where(tmp_th_dtms3 < 1.25**3, tmp_th_dtms3, 0.)) / (
            #             tmp_th_dtms3.shape[2] * tmp_th_dtms3.shape[3])

            pq_psnr = piq.psnr(sr, hr, data_range=1.)
            pq_ssim=piq.ssim(sr, hr, data_range=1.)
            total_psnr_value+=pq_psnr
            total_ssim+= pq_ssim

            # #DEM
            pq_psnr_dem = piq.psnr(sr_dem_selected, dem, data_range=1.)
            pq_ssim_dem=piq.ssim(sr_dem_selected, dem, data_range=1.)
            total_psnr_dem_value+=pq_psnr_dem
            total_ssim_dem_value+=pq_ssim_dem


        #SR
        avg_psnr_value = (total_psnr_value / (batches))
        avg_ssim_value = (total_ssim / (batches))

        #DEM
        avg_psnr_dem_value = (total_psnr_dem_value / (batches))
        avg_ssim_dem_value = (total_ssim_dem_value / (batches))

        #Absolute Relative Error
        avg_are_value = (total_absolute_relative_error / batches)
        avg_are_DTM_value = (total_absolute_relative_errorDTM / batches)

        #threshold Error
        # avg_th_values1 = (total_th_metrics1 / batches)/batch_size
        # avg_th_dtm_values1 = (total_th_dtm_metrics1 / batches)/batch_size
        # avg_th_values2 = (total_th_metrics2 / batches)/batch_size
        # avg_th_dtm_values2 = (total_th_dtm_metrics2 / batches)/batch_size
        # avg_th_values3 = (total_th_metrics3 / batches)/batch_size
        # avg_th_dtm_values3 = (total_th_dtm_metrics3 / batches)/batch_size

        #RMSE
        avg_rmse = (total_rmse / batches)
        avg_rmse_dtm = (total_rmse_dtm / batches)



        print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        print(f"avg PSNR DEM: {avg_psnr_dem_value:.3f} avg SSIM DEM: {avg_ssim_dem_value:.3f}")
        print(f"avg Absolute Error: ", avg_are_value.item())
        print(f"avg Absolute DTM Error: ", avg_are_DTM_value.item())
        print(f"avg RMSE SR/HR: ", avg_rmse.item())
        print(f"avg RMSE DTM: ", avg_rmse_dtm.item())
        #print(f"avg Threshold SR Error 1.25: ", avg_th_values1.item()," avg Threshold SR Error 1.25^2: ",avg_th_values2.item(), " avg Threshold SR Error 1.25^3: ",avg_th_values3.item())
        #print(f"avg Threshold SR DTM Error 1.25: ", avg_th_dtm_values1.item()," avg Threshold SR DTM Error 1.25^2: ",avg_th_dtm_values2.item(), " avg Threshold SR DTM Error 1.25^3: ",avg_th_dtm_values3.item())


    return avg_psnr_value, avg_psnr_dem_value


def main() -> None:
    # if pretrained:
    #     state_dict_Gen = torch.load(model_path_train)
    #     generator.load_state_dict(state_dict_Gen)

    train_dataset = BaseDataset_wth_folders_wthDEM(train_dir, image_size, upscale_factor, "train")
    valid_dataset = BaseDataset_wth_folders_wthDEM(valid_dir, image_size, upscale_factor, "validate")

    my_random_sampler_test = random.sample(range(len(valid_dataset.filenames)),int((len(valid_dataset.filenames) * 1) / 100))
    my_random_sampler_test = SubsetRandomSampler(my_random_sampler_test)

    valid_dataloader = DataLoader(valid_dataset, batch_size, False, num_workers=8, sampler=my_random_sampler_test)
    class_dict=train_dataset.a_dictionary
    best_ssimSR_value, best_ssimDEM_value, avg_psnr_values = 0.0, 0.0, 0.0


    for epoch in range(0, epochs):
        my_random_sampler = random.sample(range(len(train_dataset.filenames)),int((len(train_dataset.filenames) * 10) / 100))
        my_random_sampler = SubsetRandomSampler(my_random_sampler)
        train_dataloader = DataLoader(train_dataset, batch_size, num_workers=8, sampler=my_random_sampler)
        start=time.time()
        one_train(train_dataloader, epoch, class_dict)
        print("Time per epoch: ", time.time() - start)
        if (epoch + 1) % 1 == 0:
            psnr_value, psnr_dem_value= validate(valid_dataloader)
            best_psnr_values = (psnr_value + psnr_dem_value) / 2
            if best_psnr_values > avg_psnr_values:
                print("Epoch: ", epoch, " Best ssim values: ", best_psnr_values, " SR psnr: ", psnr_value, " DTM psnr: ", psnr_dem_value)
                torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best.pth"))
                if model_type == 'model_a' or model_type == 'model_c':
                    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-best.pth"))
                avg_psnr_values = best_psnr_values


if __name__ == "__main__":
    main()
