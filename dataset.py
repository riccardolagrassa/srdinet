import os
import random
from typing import Tuple
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from osgeo import gdal
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from config import true_super_resolution
# class BaseDataset(Dataset):
#     """The basic data set loading function only needs to prepare high-resolution image data.
#
#     Args:
#         dataroot         (str): Training data set address.
#         image_size       (int): High resolution image size.
#         upscale_factor   (int): Magnification.
#         mode             (str): Data set loading method, the training data set is for data enhancement,
#                                 and the verification data set is not for data enhancement.
#     """
#
#     def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
#         super(BaseDataset, self).__init__()
#         lr_image_size = (image_size // upscale_factor, image_size // upscale_factor)
#         hr_image_size = (image_size, image_size)
#         self.folders = [x for x in os.listdir(dataroot)]
#
#         paths = os.listdir(dataroot)
#         self.files=[]
#         for folder in paths:
#             for x in os.listdir(dataroot+folder):
#                 self.files.append(os.path.join(dataroot,folder,x))
#
#
#         # Low-resolution images and high-resolution images have different processing methods.
#         if mode == "train":
#
#             self.hr_transforms = transforms.Compose([
#                 transforms.RandomResizedCrop(hr_image_size, scale=(0.9, 1)),
#                 #transforms.RandomCrop(hr_image_size),
#                 transforms.RandomRotation(90),
#                 transforms.RandomHorizontalFlip(0.5),
#                 transforms.ToTensor()
#             ])
#
#             self.lr_transforms = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize(lr_image_size, interpolation=Image.BILINEAR),
#                 transforms.ToTensor()
#             ])
#
#         elif mode == "valid":
#
#             self.hr_transforms = transforms.Compose([
#                 transforms.Resize(hr_image_size),
#                 # transforms.CenterCrop(hr_image_size),
#                 transforms.ToTensor()
#             ])
#
#             self.lr_transforms = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize(lr_image_size, interpolation=Image.BILINEAR),
#                 transforms.ToTensor()
#             ])
#
#         elif mode == "create_lr-hr_dataset":
#             self.hr_transforms = transforms.Compose([
#                 transforms.Resize(hr_image_size),
#                 transforms.ToTensor()
#             ])
#
#
#         self.lr_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(lr_image_size, interpolation=Image.BILINEAR),
#             transforms.ToTensor()
#         ])
#
#     def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
#         hr = Image.open(self.files[index])
#         try:
#             if (hr.mode != 'RGB'):
#                 hr=hr.convert(mode='RGB')
#         except Exception as e:
#             print(e)
#         hr = self.hr_transforms(hr)
#         lr = self.lr_transforms(hr)
#
#         # HR image range is between [-1, 1].
#         hr = (hr * 2) - 1
#
#         return lr, hr
#
#     def __len__(self) -> int:
#         return len(self.files)
#
#
# class CustomDataset(Dataset):
#     """Customize the data set loading function and prepare low/high resolution image data in advance.
#
#     Args:
#         dataroot         (str): Training data set address.
#         image_size       (int): High resolution image size.
#         upscale_factor   (int): Magnification.
#         mode             (str): Data set loading method, the training data set is for data enhancement,
#                                 and the verification data set is not for data enhancement.
#     """
#
#     def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
#         super(CustomDataset, self).__init__()
#         # Get the index of all images in the high-resolution folder and low-resolution folder
#         # under the data set address.
#         # Note: The high and low resolution file index should be corresponding.
#         lr_dir_path = os.path.join(dataroot, f"LRunknownx{upscale_factor}")
#         hr_dir_path = os.path.join(dataroot, f"HR")
#         self.filenames = os.listdir(lr_dir_path)
#         self.lr_filenames = [os.path.join(lr_dir_path, x) for x in self.filenames]
#         self.hr_filenames = [os.path.join(hr_dir_path, x) for x in self.filenames]
#
#         self.image_size = image_size  # HR image size.
#         self.upscale_factor = upscale_factor
#         self.mode = mode
#
#     def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
#         lr = Image.open(self.lr_filenames[index])
#         hr = Image.open(self.hr_filenames[index])
#
#         # Data enhancement methods.
#         if self.mode == "train":
#             lr, hr = random_crop(lr, hr, self.image_size, self.upscale_factor)
#         else:
#             lr, hr = center_crop(lr, hr, self.image_size, self.upscale_factor)
#
#         # `PIL.Image` image data is converted to `Tensor` format data.
#         lr = image2tensor(lr)
#         hr = image2tensor(hr)
#
#         # HR image range is between [-1, 1].
#         hr = (hr * 2) - 1
#
#         return lr, hr
#
#     def __len__(self) -> int:
#         return len(self.filenames)
#
#
# class BaseDataset_wth_folders(Dataset):
#     def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str, input_mode: str) -> None:
#         super(BaseDataset_wth_folders, self).__init__()
#         lr_image_size = (image_size // upscale_factor, image_size // upscale_factor)
#         hr_image_size = (image_size, image_size)
#         self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
#         self.input_mode = input_mode
#         # Low-resolution images and high-resolution images have different processing methods.
#         if mode == "train":
#             self.hr_transforms = transforms.Compose([
#                 #transforms.RandomResizedCrop(hr_image_size, scale=(0.9, 1)),
#                 #transforms.RandomCrop(hr_image_size),
#                 #transforms.RandomRotation(90),
#                 #transforms.RandomHorizontalFlip(0.5),
#                 transforms.ToTensor()
#             ])
#
#             self.lr_transforms = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize(lr_image_size, interpolation=Image.BICUBIC),
#                 transforms.ToTensor()
#             ])
#
#
#
#         elif mode =='validate':
#             self.hr_transforms = transforms.Compose([
#                 #transforms.Resize(hr_image_size),
#                 #transforms.CenterCrop(hr_image_size),
#                 transforms.ToTensor()
#             ])
#
#             self.lr_transforms = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize(lr_image_size, interpolation=Image.BICUBIC),
#                 transforms.ToTensor()
#             ])
#
#
#
#     def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
#         if self.input_mode =='4c':
#             package = torch.from_numpy(numpy.load(self.filenames[index]))
#             #package = Image.open(self.filenames[index])
#             dem = package[3:4, :, :]  # because the filtered images is located into 4 channel (with cat operation)
#             rgb_selected = package[0:3, :, :]  # RGB images extracted
#             rgb_selected = transforms.ToPILImage()(rgb_selected)
#             #plt.imshow(rgb_selected)
#             #plt.show()
#
#             #dem_selected = transforms.ToPILImage()(dem_selected)
#
#
#             # try:
#             #     if (rgb_selected.mode != 'RGB'):
#             #         rgb_selected=rgb_selected.convert(mode='RGB')
#             #
#             # except Exception as e:
#             #     print(e)
#             hr = self.hr_transforms(rgb_selected)
#             lr = self.lr_transforms(hr)
#
#             return lr, hr, dem
#         else:
#             hr = Image.open(self.filenames[index])
#             # try:
#             #     if (hr.mode != 'RGB'):
#             #         hr=hr.convert(mode='RGB')
#             #
#             # except Exception as e:
#             #     print(e)
#
#
#             hr = self.hr_transforms(hr)
#             lr = self.lr_transforms(hr)
#             return lr, hr
#
#     def __len__(self) -> int:
#         return len(self.filenames)

class BaseDataset_wth_folders_wthDEM(Dataset):


    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(BaseDataset_wth_folders_wthDEM, self).__init__()
        self.lr_image_size = (image_size // upscale_factor, image_size // upscale_factor)
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
        self.mode = mode

        len_range = [i for i in numpy.arange(0, 1, 0.05)] #third elements  -> range in meters (e.g 100 meters per range)
        self.a_dictionary = dict.fromkeys(range(len(len_range) - 1))
        for idx_data in range(len(len_range) - 1):
            self.a_dictionary[idx_data] = [len_range[idx_data], len_range[idx_data + 1]]

        # Low-resolution images and high-resolution images have different processing methods.
        if self.mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.ToTensor()
            ])

            self.lr_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.lr_image_size, interpolation=Image.BICUBIC),
                transforms.PILToTensor()
            ])



        elif mode =='validate':
            self.hr_transforms = transforms.Compose([
                transforms.ToTensor()
            ])

            self.lr_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.lr_image_size, interpolation=Image.BICUBIC),
                transforms.PILToTensor()
            ])


    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
            x = gdal.Open(self.filenames[index])
            x_array = x.ReadAsArray()
            x_tensor_package = torch.FloatTensor(x_array)
            if not true_super_resolution:
                dem_source = x_tensor_package[1, :, :].unsqueeze(0)
                min_dtm_tile, max_dtm_tile = torch.min(dem_source), torch.max(dem_source)
                hr =  x_tensor_package[0, :, :].unsqueeze(0).unsqueeze(0)
                if self.mode == 'train':
                    hr = (hr - torch.min(hr)) / (torch.max(hr) - torch.min(hr))
                    dem = (dem_source - torch.min(dem_source)) / (torch.max(dem_source) - torch.min(dem_source))
                elif self.mode == 'validate':
                    hr = (hr - torch.min(hr)) / (torch.max(hr) - torch.min(hr))
                    dem = (dem_source - torch.min(dem_source)) / (torch.max(dem_source) - torch.min(dem_source))
                lr = F.interpolate(hr,self.lr_image_size,mode='bicubic')
                hr = hr.squeeze(0)
                lr = lr.squeeze(0)

                while (1):
                    if torch.isnan(dem).any() or torch.min(dem_source) < -9000:
                        random_int = random.randint(0, len(self.filenames)-1)
                        x = gdal.Open(self.filenames[random_int])
                        x_array = x.ReadAsArray()
                        x_tensor_package = torch.FloatTensor(x_array)
                        dem_source = x_tensor_package[1, :, :].unsqueeze(0)
                        min_dtm_tile, max_dtm_tile = torch.min(dem_source), torch.max(dem_source)
                        hr = x_tensor_package[0, :, :].unsqueeze(0).unsqueeze(0)
                        if self.mode == 'train':
                            hr = (hr - torch.min(hr)) / (torch.max(hr) - torch.min(hr))
                            dem = (dem_source - torch.min(dem_source)) / (torch.max(dem_source) - torch.min(dem_source))
                        elif self.mode == 'validate':
                            # for t1 in self.a_dictionary.keys():
                            #     if torch.median(dem_source) >= self.a_dictionary[t1][0] and torch.median(dem_source) < self.a_dictionary[t1][1]:
                            #         self.class_assigned = t1
                            #         break
                            hr = (hr - torch.min(hr)) / (torch.max(hr) - torch.min(hr))
                            dem = (dem_source - torch.min(dem_source)) / (torch.max(dem_source) - torch.min(dem_source))
                        lr = F.interpolate(hr, self.lr_image_size, mode='bicubic')
                        hr = hr.squeeze(0)
                        lr = lr.squeeze(0)
                    else:
                        transform = x.GetProjection()
                        return lr, hr, dem, self.filenames[index],transform, min_dtm_tile, max_dtm_tile


            elif true_super_resolution:
                hr = x_tensor_package.unsqueeze(0).unsqueeze(0)
                if self.mode == 'validate':
                    hr = (hr - torch.min(hr)) / (torch.max(hr) - torch.min(hr))
                    hr = hr.squeeze(0)
                    transform=x.GetProjection()
                    return hr, hr, hr, self.filenames[index],transform,transform,transform
                else:
                    print("Error data loader in true-resolution mode")
                    exit()

    def __len__(self) -> int:
        return len(self.filenames)