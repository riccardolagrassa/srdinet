import random
import piq
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from config_noGan import *
from dataset import BaseDataset_wth_folders_wthDEM


def one_train(train_dataloader, epoch) -> None:
    batches = len(train_dataloader)
    generator.train()
    downSNet.train()
    for index, (lr, hr, dem,tile_name) in enumerate(train_dataloader):
        lr = lr.cuda()
        hr = hr.type(lr.dtype).cuda()
        dem = dem.type(lr.dtype).cuda()

        #Generator
        g_optimizer.zero_grad()

        #NEW
        downSNet_optimizer.zero_grad()
        weight_tahn=downSNet(lr)
        new_lr = lr + weight_tahn

        sr, sr_dem_selected = generator(new_lr)
        loss =  pixel_weight * criterion(sr, hr) + dtm_weight * criterion(sr_dem_selected, dem)


        print(f"Train Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
              f"Loss: {loss.item():.6f}.")
        loss.backward()
        g_optimizer.step()
        downSNet_optimizer.step()

        scheduler.step()
        downSNet_scheduler.step()


def validate(valid_dataloader, epoch) -> float:
    batches = len(valid_dataloader)
    downSNet.eval()
    generator.eval()
    counter=0
    total_psnr_value, total_ssim, total_psnr_dem_value,total_ssim_dem_value, total_psnr_valueSR_Residual,total_ssimSR_Residual, total_psnr_valueDEM_Residual, total_ssimDEM_Residual = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for index, (lr, hr, dem, tile_name) in enumerate(valid_dataloader):
            lr = lr.cuda()
            hr = hr.cuda()
            dem = dem.cuda()

            # NEW
            weight_tahn = downSNet(lr)
            lr_new = lr + weight_tahn

            sr, sr_dem_selected = generator(lr_new)

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
        avg_psnr_value = total_psnr_value / batches
        avg_ssim_value = total_ssim / batches

        #DEM
        avg_psnr_dem_value = total_psnr_dem_value / batches
        avg_ssim_dem_value = total_ssim_dem_value / batches

        print(f"Valid stage: Epoch[{epoch + 1:04d}]")
        print(f"Avoided batch samples ", counter, " Total batches: ", batches)
        print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        print(f"avg PSNR DEM: {avg_psnr_dem_value:.3f} avg SSIM DEM: {avg_ssim_dem_value:.3f}")


    return avg_psnr_value, avg_psnr_dem_value


def main() -> None:
    # Load the dataset.
    train_dataset = BaseDataset_wth_folders_wthDEM(train_dir, image_size, upscale_factor, "train")
    valid_dataset = BaseDataset_wth_folders_wthDEM(valid_dir, image_size, upscale_factor, "validate")


    my_random_sampler=random.sample(range(len(train_dataset.filenames)), int((len(train_dataset.filenames) * 50) / 100))
    my_random_sampler=SubsetRandomSampler(my_random_sampler)

    # my_random_sampler_test = random.sample(range(len(valid_dataset.filenames)),
    #                                   int((len(valid_dataset.filenames) * 1) / 100))
    # my_random_sampler_test = SubsetRandomSampler(my_random_sampler_test)

    train_dataloader = DataLoader(train_dataset, batch_size, num_workers=8, sampler=my_random_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, num_workers=8)

    best_ssimSR_value, best_ssimDEM_value, avg_psnr_values = 0.0, 0.0, 0.0


    for epoch in range(start_epoch, epochs):
        one_train(train_dataloader, epoch)
        if (epoch + 1) % 5 == 0:
            psnr_value, psnr_dem_value= validate(valid_dataloader, epoch)
            best_psnr_values = (psnr_value + psnr_dem_value) / 2
            if best_psnr_values > avg_psnr_values:
                print("Epoch: ", epoch, " Best ssim values: ", best_psnr_values, " SR psnr: ", psnr_value, " DEM psnr: ", psnr_dem_value)
                torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best.pth"))
                torch.save(downSNet.state_dict(), os.path.join(exp_dir2, "downSNet-best.pth"))
                avg_psnr_values = best_psnr_values



if __name__ == "__main__":
    main()
