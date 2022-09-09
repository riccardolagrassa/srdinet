import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


torch.cuda.manual_seed_all(0)                       # Set random seed.
upscale_factor   = 4                       # How many times the size of the high-resolution image in the data set is than the low-resolution image.
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

##
# model a RLNet+DualSRGAN
# model b RLNet+DualSRNoGan
# model c our DualSRGAN
##
model_type='model_a'

if torch.cuda.is_available():
    cudnn.enabled = True
    cudnn.benchmark  = True                    # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
else:
    print("Error in GPU.")
    exit()

mode             = "validate"                 # Run mode. Specific mode loads specific variables.
true_super_resolution= True
exp_name         = "DTM_exp_train10%_"+model_type             # Experiment name.
working_dir = '/home/super/rlagrassa/DtmESRGAN/' #! Change with your path

image_size = 512  # High-resolution image size in the training dataset.
# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    train_dir = "/home/supreme/datasets-nas/hirise_dataset/train/"
    valid_dir = "/home/supreme/datasets-nas/hirise_dataset/test/"

    batch_size = 1 # Data batch size.


    if model_type == 'model_a':
        from modelNetA import Discriminator, Generator
        discriminator = Discriminator()
        generator = Generator()
    if model_type == 'model_b':
        from modelNetB import Generator
        generator = Generator()
    if model_type == 'model_c':
        from modelNetC import Discriminator, Generator
        discriminator = Discriminator()
        generator = Generator()

    # Resume training.
    #pretrained = True
    #model_path_train = working_dir+"results/"+exp_name+'/'+"g-best.pth"

    # Train epochs.
    epochs                = 50                   # The total number of cycles in the training phase of the adversarial network.


    # Loss function.
    criterion       = nn.BCELoss()      # Pixel loss.
    if model_type == 'model_a' or model_type == 'model_c':
        adversarial_criterion = nn.BCEWithLogitsLoss()     # Adversarial loss.
    pixel_weight          = 1.
    dtm_weight        = 1.
    adversarial_weight    = 0.0005

    # Optimizer.
    g_optimizer           = optim.Adam(generator.parameters(), lr=0.0001)  # Generator learning rate during adversarial network training.
    scheduler = CosineAnnealingLR(g_optimizer, epochs)

    if model_type == 'model_a' or model_type == 'model_c':
        d_optimizer           = optim.Adam(discriminator.parameters(), lr=0.0001)  # Discriminator learning rate during adversarial network training.
        d_scheduler = CosineAnnealingLR(d_optimizer, epochs)
        discriminator = nn.DataParallel(discriminator.cuda())

    warmup_epochs = 0


    generator = nn.DataParallel(generator.cuda())

    if not os.path.exists(working_dir+'results/'):
        os.makedirs(working_dir+'results/')

    if not os.path.exists(working_dir+'results/'+exp_name):
        os.makedirs(working_dir+'results/'+exp_name)

    exp_dir2              = working_dir + 'results/' + exp_name

# ==============================================================================
#                              Validate configure
# ==============================================================================

if mode == "validate":
    print("Model selected: ", model_type)
    # Additional variables.
    exp_dir    = os.path.join("results", "test", exp_name)
    batch_size = 1  # Data batch size.

    # Load model.
    if model_type == 'model_a':
        from modelNetA import Generator
        generator = Generator()
    if model_type == 'model_b':
        from modelNetB import Generator
        generator = Generator()
    if model_type == 'model_c':
        from modelNetC import Generator
        generator = Generator()

    generator = nn.DataParallel(generator).cuda()
    model_path_train = working_dir+"results/"+exp_name+'/'+"g-best.pth"
    pytorch_total_params = sum(p.numel() for p in generator.parameters())
    print("Model's Parameter: ", pytorch_total_params)

    # Test data address.
    valid_dir = "/home/super/tmp_datasets/500_hirise_instances_GT/test/" #only 500 instances GT

    #Directories name (change names)
    sr_dir = working_dir+"results/test/"+exp_name+'RealGTInput='+str(true_super_resolution) + '/' + 'sr_images'  # Super-resolution image address.
    sr_extra_dir = working_dir+"results/test/" + exp_name+'RealGTInput='+str(true_super_resolution)  + '/' + 'sr_dem_images'  # Super-resolution image address.
    hr_extra_dir = working_dir+"results/test/"+exp_name+'RealGTInput='+str(true_super_resolution) + '/' +'hr_dem_images'  # Super-resolution image address.
    hr_dir = working_dir+"results/test/"+exp_name+'RealGTInput='+str(true_super_resolution) +'/' +'hr_images'  # Super-resolution image address.
    lr_dir = working_dir+"results/test/"+exp_name+'RealGTInput='+str(true_super_resolution) +'/' +'lr_images'  # Super-resolution image address.
    if model_type == 'model_a' or model_type == 'model_b':
        rlr_dir = working_dir + "results/test/" + exp_name+'RealGTInput='+str(true_super_resolution)  + '/' + 'rlr_images'  # Super-resolution image address.
