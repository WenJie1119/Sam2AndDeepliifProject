import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

class InferenceTiler:
    """
    Iterable class to tile image(s) and stitch result tiles together.
    Based on DeepLIIF original implementation.
    """
    def __init__(self, orig, tile_size, overlap_size=0, pad_size=0, pad_color=(255, 255, 255)):
        if tile_size <= 0:
            raise ValueError('InferenceTiler input tile_size must be positive and non-zero')
        if overlap_size < 0:
            raise ValueError('InferenceTiler input overlap_size must be positive or zero')
        if pad_size < 0:
            raise ValueError('InferenceTiler input pad_size must be positive or zero')

        self.single_orig = not isinstance(orig, list)
        if self.single_orig:
            orig = [orig]

        for i in range(1, len(orig)):
            if orig[i].size != orig[0].size:
                raise ValueError('InferenceTiler input images do not have the same size.')
        
        self.orig_width = orig[0].width
        self.orig_height = orig[0].height

        patch_size = tile_size - (2 * pad_size)

        # Pad images if smaller than patch size
        if orig[0].width < patch_size:
            for i in range(len(orig)):
                while orig[i].width < patch_size:
                    mirrored = ImageOps.mirror(orig[i])
                    orig[i] = ImageOps.expand(orig[i], (0, 0, orig[i].width, 0))
                    orig[i].paste(mirrored, (mirrored.width, 0))
                orig[i] = orig[i].crop((0, 0, patch_size, orig[i].height))
        
        if orig[0].height < patch_size:
            for i in range(len(orig)):
                while orig[i].height < patch_size:
                    flipped = ImageOps.flip(orig[i])
                    orig[i] = ImageOps.expand(orig[i], (0, 0, 0, orig[i].height))
                    orig[i].paste(flipped, (0, flipped.height))
                orig[i] = orig[i].crop((0, 0, orig[i].width, patch_size))

        self.image_width = orig[0].width
        self.image_height = orig[0].height

        overlap_width = 0 if patch_size >= self.image_width else overlap_size
        overlap_height = 0 if patch_size >= self.image_height else overlap_size
        
        center_width = patch_size - (2 * overlap_width)
        center_height = patch_size - (2 * overlap_height)
        
        if center_width <= 0 or center_height <= 0:
            raise ValueError('InferenceTiler combined overlap_size and pad_size are too large')

        self.c0x = pad_size
        self.c0y = pad_size
        self.c1x = overlap_width + pad_size
        self.c1y = overlap_height + pad_size
        self.c2x = patch_size - overlap_width + pad_size
        self.c2y = patch_size - overlap_height + pad_size
        self.c3x = patch_size + pad_size
        self.c3y = patch_size + pad_size
        self.p1x = overlap_width
        self.p1y = overlap_height
        self.p2x = patch_size - overlap_width
        self.p2y = patch_size - overlap_height

        self.overlap_width = overlap_width
        self.overlap_height = overlap_height
        self.patch_size = patch_size
        self.center_width = center_width
        self.center_height = center_height

        self.orig = orig
        self.tile_size = tile_size
        self.pad_size = pad_size
        self.pad_color = pad_color
        self.res = {}

    def __iter__(self):
        for y in range(0, self.image_height, self.center_height):
            for x in range(0, self.image_width, self.center_width):
                if x + self.patch_size > self.image_width:
                    x = self.image_width - self.patch_size
                if y + self.patch_size > self.image_height:
                    y = self.image_height - self.patch_size
                self.x = x
                self.y = y
                tiles = [im.crop((x, y, x + self.patch_size, y + self.patch_size)) for im in self.orig]
                if self.pad_size != 0:
                    tiles = [ImageOps.expand(t, self.pad_size, self.pad_color) for t in tiles]
                yield tiles[0] if self.single_orig else tiles

    def stitch(self, result_tiles):
        for k, tile in result_tiles.items():
            if k not in self.res:
                self.res[k] = Image.new('RGB', (self.image_width, self.image_height))
            if tile.size != (self.tile_size, self.tile_size):
                tile = tile.resize((self.tile_size, self.tile_size))
            
            self.res[k].paste(tile.crop((self.c1x, self.c1y, self.c2x, self.c2y)), (self.x + self.p1x, self.y + self.p1y))

            if self.x == 0 and self.y == 0:
                 self.res[k].paste(tile.crop((self.c0x, self.c0y, self.c1x, self.c1y)), (self.x, self.y))
            if self.y == 0:
                self.res[k].paste(tile.crop((self.c1x, self.c0y, self.c2x, self.c1y)), (self.x + self.p1x, self.y))
            if self.x == self.image_width - self.patch_size and self.y == 0:
                self.res[k].paste(tile.crop((self.c2x, self.c0y, self.c3x, self.c1y)), (self.x + self.p2x, self.y))
            if self.x == 0:
                self.res[k].paste(tile.crop((self.c0x, self.c1y, self.c1x, self.c2y)), (self.x, self.y + self.p1y))
            if self.x == self.image_width - self.patch_size:
                self.res[k].paste(tile.crop((self.c2x, self.c1y, self.c3x, self.c2y)), (self.x + self.p2x, self.y + self.p1y))
            if self.x == 0 and self.y == self.image_height - self.patch_size:
                self.res[k].paste(tile.crop((self.c0x, self.c2y, self.c1x, self.c3y)), (self.x, self.y + self.p2y))
            if self.y == self.image_height - self.patch_size:
                self.res[k].paste(tile.crop((self.c1x, self.c2y, self.c2x, self.c3y)), (self.x + self.p1x, self.y + self.p2y))
            if self.x == self.image_width - self.patch_size and self.y == self.image_height - self.patch_size:
                self.res[k].paste(tile.crop((self.c2x, self.c2y, self.c3x, self.c3y)), (self.x + self.p2x, self.y + self.p2y))

    def results(self):
        if self.orig_width != self.image_width or self.orig_height != self.image_height:
            return {k: im.crop((0, 0, self.orig_width, self.orig_height)) for k, im in self.res.items()}
        else:
            return {k: im for k, im in self.res.items()}

def disable_batchnorm_tracking_stats(model):
    """
    Disable batchnorm tracking stats for inference.
    Required for consistency with DeepLIIF original results.
    """
    for m in model.modules():
        for child in m.children():
            if type(child) == torch.nn.BatchNorm2d:
                child.track_running_stats = False
                child.running_mean_backup = child.running_mean
                child.running_mean = None
                child.running_var_backup = child.running_var
                child.running_var = None
    return model

def make_power_2(img, base=4, method=Image.BICUBIC):
    """Adjust image size to be a multiple of base."""
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)

def get_transform():
    """Returns the standard DeepLIIF image transformation (including batch dimension)."""
    return transforms.Compose([
        transforms.Lambda(lambda i: make_power_2(i, base=4, method=Image.BICUBIC)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda t: t.unsqueeze(0))
    ])

def tensor_to_pil(tensor):
    """Convert normalized PyTorch tensor (-1 to 1) to PIL Image.
    Matches original DeepLIIF tensor2im function exactly.
    """
    if not isinstance(tensor, np.ndarray):
        if isinstance(tensor, torch.Tensor):
            image_tensor = tensor.data
        else:
            return tensor
        # Use [0] to get first sample, exactly like original DeepLIIF
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: transpose and scaling (CHW -> HWC, denormalize)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = tensor
    return Image.fromarray(image_numpy.astype(np.uint8))
