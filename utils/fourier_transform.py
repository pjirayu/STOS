"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""

#--------------------------
# Fourier Domain Adaptation
#Citation (main): https://github.com/YanchaoYang/FDA/blob/b9a0cdf0bcc9b787c00e39df73eda5673706f219/train.py#L77
#Citation (fn): https://github.com/YanchaoYang/FDA/blob/b9a0cdf0bcc9b787c00e39df73eda5673706f219/utils/__init__.py#L38
#--------------------------

import numpy as np
import os
import tqdm
import random
from PIL import Image
from typing import Optional, Sequence
import torch.nn as nn
import torch
import torch.fft

#---for FFT tensor---
'''
def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    print(fft_im.shape, type(fft_im))
    fft_amp = fft_im[:,:,0]**2 + fft_im[:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,1], fft_im[:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    #fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    #fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )
    fft_src = torch.fft.rfft2( src_img.clone() ) 
    fft_trg = torch.fft.rfft2( trg_img.clone() )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    #src_in_trg = torch.fft.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    src_in_trg = torch.fft.irfft2( fft_src_, s=[imgH,imgW] )

    return src_in_trg
'''

#---for FFT array---
def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

#---L = beta of fourier transforming---
def FDA_source_to_target_np( src_img, trg_img, beta=0.01 ):

    src_img = np.squeeze(src_img)
    trg_img = np.squeeze(trg_img)

    if trg_img.shape != src_img.shape:
        raise ValueError(
            "The source and target images must have the same shape,"
            " but got {} and {} respectively.".format(src_img.shape, trg_img.shape)
        )

    # exchange magnitude
    # input: src_img, trg_img --> ndarray type

    #src_img_np = src_img #.cpu().numpy()
    #trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    #fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    #fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    fft_src_np = np.fft.fft2( src_img.astype(np.float32), axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img.astype(np.float32), axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    #amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)
    amp_trg = np.abs(fft_trg_np)

    # mutate the amplitude part of source with target
    #function
    #amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    #directly
    amp_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    amp_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    height, width = amp_src.shape[:2]
    border = np.floor(min(height, width) * beta).astype(int)
    center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amp_src[y1:y2, x1:x2] = amp_trg[y1:y2, x1:x2]
    amp_src = np.fft.ifftshift(amp_src, axes=(-2, -1))

    # mutated fft of source
    #fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    ## get the mutated image
    #src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    #src_in_trg = np.real(src_in_trg)

    # get mutated image
    src_image_transformed = np.fft.ifft2(amp_src * np.exp(1j * pha_src), axes=(-2, -1))
    src_image_transformed = np.real(src_image_transformed)

    #from https://github.com/thuml/Transfer-Learning-Library/blob/112fc9ff7c420d0717032770cac61b79cd8b7724/dalib/translation/fourier_transform.py#L122
    src_as_trg_img = src_image_transformed.transpose((1, 2, 0))  # C,H,W --> H,W,C
    #print(src_as_trg_img.shape)
    src_as_trg_img = Image.fromarray(src_as_trg_img.astype('uint8')).convert('RGB').save('demo_images/src_as_trg_img.png') #.clip(min=0, max=255).astype('uint8')).convert('RGB')

    return src_image_transformed, src_as_trg_img


#----------------
#Original by https://github.com/albumentations-team/albumentations/blob/477156d5f14a26f8f0f71c5f2c0d59748bb7de4b/albumentations/augmentations/domain_adaptation.py#L32
#----------------
'''
def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray, beta: float) -> np.ndarray:
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
    Args:
        img:  source image
        target_img:  target image for domain adaptation
        beta: coefficient from source paper
    Returns:
        transformed image
    """

    img = np.squeeze(img)
    target_img = np.squeeze(target_img)

    if target_img.shape != img.shape:
        raise ValueError(
            "The source and target images must have the same shape,"
            " but got {} and {} respectively.".format(img.shape, target_img.shape)
        )

    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    fft_trg = np.fft.fft2(target_img.astype(np.float32), axes=(0, 1))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src, axes=(0, 1))
    amplitude_trg = np.fft.fftshift(amplitude_trg, axes=(0, 1))
    height, width = amplitude_src.shape[:2]
    border = np.floor(min(height, width) * beta).astype(int)
    center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]
    amplitude_src = np.fft.ifftshift(amplitude_src, axes=(0, 1))

    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))
    src_image_transformed = np.real(src_image_transformed)

    return src_image_transformed


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    @property
    def targets(self):
        return {"image": self.apply}


class FDA(ImageOnlyTransform):
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
    Simple "style transfer".
    Args:
        reference_images (List[str] or List(np.ndarray)): List of file paths for reference images
            or list of reference images.
        beta_limit (float or tuple of float): coefficient beta from paper. Recommended less 0.3.
        read_fn (Callable): Used-defined function to read image. Function should get image path and return numpy
            array of image pixels.
    Targets:
        image
    Image types:
        uint8, float32
    Reference:
        https://github.com/YanchaoYang/FDA
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf
    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> target_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> aug = A.Compose([A.FDA([target_image], p=1, read_fn=lambda x: x)])
        >>> result = aug(image=image)
    """

    def __init__(
        self,
        reference_images: List[Union[str, np.ndarray]],
        beta_limit=0.1,
        read_fn=read_rgb_image,
        always_apply=False,
        p=0.5,
    ):
        super(FDA, self).__init__(always_apply=always_apply, p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.beta_limit = to_tuple(beta_limit, low=0)

    def apply(self, img, target_image=None, beta=0.1, **params):
        return fourier_domain_adaptation(img=img, target_img=target_image, beta=beta)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        target_img = self.read_fn(random.choice(self.reference_images))
        target_img = cv2.resize(target_img, dsize=(img.shape[1], img.shape[0]))

        return {"target_image": target_img}

    def get_params(self):
        return {"beta": random.uniform(self.beta_limit[0], self.beta_limit[1])}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("reference_images", "beta_limit", "read_fn")

    def _to_dict(self):
        raise NotImplementedError("FDA can not be serialized.")
'''