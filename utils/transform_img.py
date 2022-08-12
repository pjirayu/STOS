import cv2
import numpy as np
import torch

#---CLAHE transform---
def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


#---blend image---
def mix_pixel(pix_1, pix_2, perc):
    return (perc/255 * pix_1) + ((255 - perc)/255 * pix_2)

def blend_images_using_mask(img_orig, img_for_overlay, img_mask):
    if len(img_mask.shape) != 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_res = mix_pixel(img_orig, img_for_overlay, img_mask)
    return img_res.astype(np.uint8)


def blending_images(x1: np.ndarray, x2: np.ndarray, alpha: np.float32):
    return np.uint8((alpha * x1) + ((1.-alpha) * x2))

def blending_images_cv2(src1, src2, alpha):
    return  cv2.addWeighted(src1, alpha, src2, 1.-alpha, 0.0)

#---ZCA image whitening (prototype)---
class ZCATransformation(object):
    def __init__(self, transformation_matrix, transformation_mean):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix
        self.transformation_mean = transformation_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(1) * tensor.size(2) * tensor.size(3) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor[0].size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        batch = tensor.size(0)

        flat_tensor = tensor.view(batch, -1)
        transformed_tensor = torch.mm(flat_tensor - self.transformation_mean, self.transformation_matrix)

        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string

#---zca simplified---
#batchrun
def zca_batch(x):
    """Computes ZCA transformation for the dataset.
    Args:
        x: dataset.
    Returns:
        ZCA transformation matrix and mean matrix.
    """
    [B, C, H, W] = list(x.size())
    x = x.reshape((B, C*H*W))       # flatten the data
    x = x - torch.mean(x, dim=0, keepdim=True)             
    #covariance = torch.matmul(x.transpose(0, 1), x) / B
    covariance = x.t() @ x
    #U, S, V = np.linalg.svd(covariance.cpu().detach())
    U, S, V = torch.linalg.svd(covariance)
    eps = 1e-1
    #W = np.matmul(np.matmul(U, np.diag(1. / np.sqrt(S + eps))), U.T)
    W = torch.matmul(torch.matmul(U, torch.diag(1. / torch.sqrt(S + eps))), U.T)
    #return torch.Tensor(W)
    return W

#single image --- Unable to bear on overload processing, may crash the memory usage--- 
def zca_image(x):
    """Computes ZCA transformation for the dataset.
    Args:
        x: dataset.
    Returns:
        ZCA transformation matrix and mean matrix.
    """
    #[C, H, W] = list(x.size())
    C, H, W = x.shape
    x = x.reshape(C*H*W)       # flatten the data
    mean = np.mean(x, axis=0, keepdims=True)
    x -= mean
    covariance = x.reshape(-1, 1) @ x.reshape(1, -1)
    U, S, V = np.linalg.svd(covariance)
    eps = 1e-3
    W = np.matmul(np.matmul(U, np.diag(1. / np.sqrt(S + eps))), U.T)
    x_whiten = W @ x
    return x_whiten

#---exemplary application---
#image-level
#W1= transform_img.zca_batch(X1)
#W2= transform_img.zca_batch(X2)
#X1, X2 = torch.matmul(X1[0:opt['mini_batch_size_g_h']].reshape((31, 3*224*224)), W1),\
#    torch.matmul(X2[0:opt['mini_batch_size_g_h']].reshape((31, 3*224*224)), W2)
#X1, X2 = X1.reshape((31, 3,224,224)), X2.reshape((31, 3,224,224))

#feature-level
#W1 = transform_img.zca_batch(encoder_X1)
#W2 = transform_img.zca_batch(encoder_X2)
#X1 = torch.matmul(encoder_X1, W1)
#X2 = torch.matmul(encoder_X2, W2)