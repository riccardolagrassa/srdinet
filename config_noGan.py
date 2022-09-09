import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from model_without_discriminator import Generator, DownSamplingNetwork
from modelNetA import Generator, DownSamplingNetwork


torch.cuda.manual_seed_all(0)                       # Set random seed.
upscale_factor   = 4                       # How many times the size of the high-resolution image in the data set is than the low-resolution image.
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if torch.cuda.is_available():
    cudnn.enabled = True
    cudnn.benchmark  = True                    # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
else:
    print("Error in GPU.")
    exit()

mode             = "train"                 # Run mode. Specific mode loads specific variables.
exp_name         = "DTM_ESGRAN_NOGAN_lessepoches"              # Experiment name.
working_dir = '/home/super/rlagrassa/DtmESRGAN/'

#DTM_ESGRAN_NOGAN
image_size = 512  # High-resolution image size in the training dataset.
# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    train_dir = "/home/supreme/datasets-nas/hirise_dataset/train/"
    valid_dir = "/home/supreme/datasets-nas/hirise_dataset/test/"

    batch_size = 3 # Data batch size.

    generator             = Generator()     # Load the generator model.
    downSNet              = DownSamplingNetwork()

    # Resume training.
    start_p_epoch         = 0                           # The number of initial iterations of the generator training phase. When set to 0, it means incremental training.
    start_epoch           = 0                           # The number of initial iterations of the adversarial training phase. When set to 0, it means incremental training.
    resume                = False                       # Set to `True` to continue training from the previous training progress.
    resume_p_weight       = ""                          # Restore the weight of the generator model during generator training.
    resume_d_weight       = ""                          # Restore the weight of the generator model during the training of the adversarial network.
    resume_g_weight       = ""                          # Restore the weight of the discriminator model during the training of the adversarial network.

    # Train epochs.
    epochs                = 20                         # The total number of cycles in the training phase of the adversarial network.


    # Loss function.
    criterion       = nn.BCELoss()      # Pixel loss.
    pixel_weight          = 0.1
    dtm_weight        = 0.1

    # Optimizer.
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)  # Generator learning rate during adversarial network training.
    downSNet_optimizer = optim.Adam(downSNet.parameters(), lr=0.0001)

    scheduler = CosineAnnealingLR(g_optimizer, epochs)
    downSNet_scheduler = CosineAnnealingLR(downSNet_optimizer, epochs)

    generator = nn.DataParallel(generator.cuda())
    downSNet = nn.DataParallel(downSNet.cuda())

    if not os.path.exists(working_dir+'results/'):
        os.makedirs(working_dir+'results/')

    if not os.path.exists(working_dir+'results/'+exp_name):
        os.makedirs(working_dir+'results/'+exp_name)

    exp_dir2 = working_dir + 'results/' + exp_name

    #print("Train rgb dataset: ", train_rgb_dir, " Train depth dataset: ", train_depth_dir)
    print("Epoches ",epochs)
    print("Pixel weight: ", pixel_weight, " depth weight: ", dtm_weight)
    print("Image size HR ", image_size)

# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "validate":
    # Additional variables.
    exp_dir    = os.path.join("results", "test", exp_name)
    batch_size = 1  # Data batch size.

    # Load model.
    generator      = Generator().cuda()
    downSNet              = DownSamplingNetwork()

    generator = nn.DataParallel(generator)
    downSNet = nn.DataParallel(downSNet)
    model_path_train = working_dir+"results/"+exp_name+'/'+"g-best.pth"
    model_path_SNet = working_dir+"results/"+exp_name+'/'+"downSNet-best.pth"


    # Test data address.
    valid_dir = "/home/supreme/datasets-nas/hirise_dataset/test/"

    sr_dir = working_dir+"results/test/"+exp_name+ '/' + 'sr_images'  # Super-resolution image address.
    sr_extra_dir = working_dir+"results/test/" + exp_name + '/' + 'sr_dem_images'  # Super-resolution image address.
    hr_extra_dir = working_dir+"results/test/"+exp_name+ '/' +'hr_dem_images'  # Super-resolution image address.
    hr_dir = working_dir+"results/test/"+exp_name+'/' +'hr_images'  # Super-resolution image address.
    lr_dir = working_dir+"results/test/"+exp_name+'/' +'lr_images'  # Super-resolution image address.
