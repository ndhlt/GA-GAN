import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from lpips import LPIPS

# FID, KID, LPIPSの計算
def calculate_metrics(real_images, generated_images):
    fid = FrechetInceptionDistance()
    fid_score = fid(real_images, generated_images)
    
    kid = KernelInceptionDistance(subset_size=100)
    kid_score = kid(real_images, generated_images)
    
    lpips_model = LPIPS(net='alex')
    lpips_score = lpips_model(real_images, generated_images)
    
    return fid_score, kid_score, lpips_score