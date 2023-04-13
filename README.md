# SRDiNet
# Abstract
In computer vision, stereoscopy allows the three-dimensional reconstruction of the scene using two 2D images taken from two slightly different points of view, to extract spatial information on the depth of the scene in the form of a map of disparities.
In stereophotogrammetry, the disparity map is essential to extract the Digital Terrain Model (DTM) and thus obtain a 3D spatial mapping necessary for a better analysis of planetary surfaces.
However, the entire reconstruction process performed with the stereo matching algorithm can be time-consuming and generate many artefacts. Coupled with the lack of adequate stereo coverage, it can pose a significant obstacle to 3D planetary mapping.
Recently, many deep learning architectures have been proposed for monocular depth estimation which aspires to predict the third dimension given a single 2D image, with considerable advantages thanks to the simplification of the reconstruction problem, leading to a significant increase in interest in deep models for the generation of super-resolution images and DTM estimation.
In this paper, we combine these last two concepts into a single end-to-end model and introduce a new Generative Adversarial Network solution that estimates the DTM at 4x resolution from a single monocular image, called SRDiNet (Super-Resolution Depth Image Network).
Furthermore, we introduce a sub-network able to apply a refinement using interpolated input images to better enhance the fine details of the final product and we demonstrated the effectiveness of its benefits through three different versions of the proposal: SRDiNet with GAN approach, SRDiNet without adversarial network and SRDiNet without the refinement learned network plus GAN approach.
The results of Oxia Planum (the landing site of the European Space Agency’s Rosalind Franklin ExoMars rover 2023) are reported applying the best model along all Oxia Planum tiles and releasing a 3D product enhanced by 4x.
The visualization tool is available at https://huggingface.co/spaces/ARTeLab/DTM_Estimation_SRandD.

#Setting config.py
models list:
* 'model_a' -> RLNet+DualSRGAN
* 'model_b' -> RLNet+DualSRNoGan
* 'model_c' -> No RLNet (DualSRGAN)
#example
#config.py
model_type='model_a'

mode             = "validate"                 # Run mode. train/validate

true_super_resolution= True  -> set mode=validate and true super-resolution=True to test instances with only gray-scale input (No tensor with double channel is due)

exp_name         = "DTM_exp_train10%_"+model_type             # Experiment name.

working_dir = 'your path/' -> Change with your path
