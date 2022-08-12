import torch
import PIL

def quantize(image, palette):
    """
    Similar to PIL.Image.quantize() in PyTorch. Built to maintain gradient.
    Only works for one image i.e. CHW. Does NOT work for batches.

    Arg:
        palette : {colors_normalized = colors/255}
        image   : one image without batch
        return    new quantized image w/o batch including
    """

    C, H, W = image.shape
    n_colors = len(palette)

    # Easier to work with list of colors
    flat_img = image.view(C, -1).T # [C, H, W] -> [H*W, C]

    # Repeat image so that there are n_color number of columns of the same image
    flat_img_per_color = torch.stack(n_colors*[flat_img], dim=-2) # [H*W, C] -> [H*W, n_colors, C]

    # Get euclidian distance between each pixel in each column and the columns repsective color
    # i.e. column 1 lists distance of each pixel to color #1 in palette, column 2 to color #2 etc.
    squared_distance = (flat_img_per_color-palette)**2
    # Dirty cursed hack
    # https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702/4
    euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=-1) + 1e-8) # [H*W, n_colors, C] -> [H*W, n_colors]


    # Get the shortest distance (one value per row (H*W) is selected)
    min_distances = torch.min(euclidean_distance, dim=-1).values # [H*W, n_colors] -> [H*W]

    # Get difference between each distance and the shortest distance.
    # One value per column (the selected value) will become 0.
    per_color_difference = euclidean_distance - torch.stack(n_colors*[min_distances], dim=-1) # [H*W, n_colors]

    # Round all values up and invert. Creates something similar to one-hot encoding.
    per_color_diff_scaled = 1 - torch.ceil(per_color_difference)

    per_color_diff_scaled_ = per_color_diff_scaled #(per_color_diff_scaled.T / per_color_diff_scaled.sum(dim=-1)).T

    # Multiply the "kinda" one-hot encoded per_color_diff_scaled with the palette colors.
    # The result is a quantized image.
    quantized = torch.matmul(per_color_diff_scaled_, palette)

    # Reshape it back to the original input format.
    quantized_img = quantized.T.view(C, H, W) # [H*W, C] -> [C, H, W]

    return quantized_img