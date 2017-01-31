import skimage.transform as ski_trs
import skimage.exposure as ski_exs
import skimage.util as ski_util
import skimage.draw as ski_draw
import numpy as np
import cv2 as cv
import numpy.random as rnd
import time

ROW = 0
COL = 1
IMAGE_Y_LEN = 32
IMAGE_X_LEN = 32
IMAGE_SHAPE = (IMAGE_Y_LEN, IMAGE_X_LEN)
DEFAULT_SHIFT_RANGE = ((-6,6), (-6,6))
DEFAULT_ANGLE_RANGE = (-50,50)
DEFAULT_SCALE_RANGE = (0.6, 1.4)
INTENCITY_DISTANCE = 90
DEFAULT_INTENSITY_RANGE = (0, 255)
DEFAULT_SHADOW_LOW = -30
DEFAULT_SHADOW_HIGH = 30
DEFAULT_X_RANGE = (-5, 5)
DEFAULT_Y_RANGE = (-5, 5)
DEFAULT_MODES = ['constant', 'edge', 'symmetric', 'reflect', 'wrap']
DEFAULT_NOIZE_MODES = ['gaussian', 'speckle']
#DEFAULT_NOIZE_MODES = ['gaussian', 'speckle', 'poisson']

ORIGIN = (0, 0)

def set_seed():
    seed = round(time.time() * 1e6)
    rnd.seed(seed & 0xFFFFFFFF)

def log(*msgs):
    print(*msgs)

def flip_coin():
    return bool(rnd.randint(0, 2))

def choose(choices, random_size=False, replace=False, at_least=3):
    if random_size:
        min_num = at_least if len(choices) > at_least else 1
        size = choose_int(min_num, len(choices))
        return rnd.choice(choices, size=size, replace=replace)
    return rnd.choice(choices)

def choose_mode():
    return rnd.choice(DEFAULT_MODES)

def choose_int(*args, size=1):
    assert(len(args) == 2)
    if size == 1:
        return rnd.randint(low=args[0], high=args[1]+1)
    return rnd.randint(low=args[0], high=args[1]+1, size=size)

def clip(a, a_min=0, a_max=255):
    return np.clip(a, a_min, a_max)

def choose_float(*args, size=1):
    assert(len(args) == 2)
    return rnd.uniform(low=args[0], high=args[1], size=size)

def flipup(img):
    #log("Flipup")
    return np.flipud(img)

def fliplr(img):
    #log("Fliplr")
    return np.fliplr(img)

def rotate(img, range_angles=DEFAULT_ANGLE_RANGE, mode=None, angle=None):
    #log("Rotate")
    if angle == None:
       assert(len(range_angles) == 2)
       angle = choose_int(*range_angles)
    if mode is None:
        mode = choose_mode()
    assert(mode in DEFAULT_MODES)
    #log("Angle=", angle)
    rotated = ski_trs.rotate(img, angle=angle, mode=mode, preserve_range=True)
    return np.uint8(rotated)

def translate(img, range_shift=DEFAULT_SHIFT_RANGE, mode=None):
    #log("Translate")
    dx = choose_int(*range_shift[1])
    dy = choose_int(*range_shift[0])
    affine = ski_trs.AffineTransform(translation=(dx, dy))
    if mode is None:
        mode = choose_mode()
    assert(mode in DEFAULT_MODES)
    translated = ski_trs.warp(img, affine, mode=mode, preserve_range=True)
    return np.uint8(translated)

def pad(img, tshape, mode):
    oshape = img.shape
    pad_fun = (lambda i:
                 ((tshape[i] - oshape[i]) // 2,
                  (tshape[i] - oshape[i]) // 2 +
                   (tshape[i] - oshape[i]) % 2))
    x_pad, y_pad = (0, 0), (0, 0)
    if oshape[ROW] < tshape[ROW]:
        y_pad = pad_fun(ROW)
    if oshape[COL] < tshape[COL]:
        x_pad = pad_fun(COL)
    pads = (y_pad, x_pad, ORIGIN)
    return ski_util.pad(img, mode=mode, pad_width=pads)

def crop(img, tshape):
    oshape = img.shape
    wy, wx = [0, 0], [0, 0]
    if oshape[ROW] > tshape[ROW]:
        range_y = (0, oshape[ROW] - tshape[ROW])
        wy[0] = choose_int(*range_y)
        wy[1] = oshape[ROW] - wy[0] - tshape[ROW] 
    if oshape[COL] > tshape[COL]:
        range_x = (0, oshape[COL] - tshape[COL])
        wx[0] = choose_int(*range_x)
        wx[1] = oshape[COL] - wx[0] - tshape[COL]
    crop = (tuple(wy), tuple(wx), ORIGIN)
    return ski_util.crop(img, crop_width=crop)

def scale(img, range_factor=DEFAULT_SCALE_RANGE, mode=None, factor=None):
    #log("Scale")
    assert(len(range_factor) == 2)
    if factor == None:
        factor = (choose_float(*range_factor), choose_float(*range_factor))
        if factor == (1.0, 1.0):
            i, j = int(flip_coin()), int(flip_coin())
            factor = (range_factor[i], range_factor[j])
    scaled_img = ski_trs.rescale(img, scale=factor, preserve_range=True)
    scaled_img = np.uint8(scaled_img)
    oshape = img.shape
    if mode is None:
        mode = choose_mode()
    scaled_img = pad(scaled_img, oshape, mode)
    scaled_img = crop(scaled_img, oshape)
    assert(scaled_img.shape == img.shape)
    return scaled_img

def shadow(img, mode=None, max_shadow=DEFAULT_SHADOW_HIGH):
    pass
    #log("Shadow")
    #shape = img.shape
    #sy, sx = shape[0], shape[1]
    #choose_side = lambda: choose([0, sx-1])
    #rand_x = lambda: np.clip(rnd.normal(loc=sx//2, scale=sx//4), 0, sx-1)
    #xo, xu, xb = choose_side(), rand_x(), rand_x()
    #yu, yb = 0, sy-1
    #r = np.array([yu, yu, yb, yb, yu])
    #c = np.array([xo, xu, xb, xo, xo])
    #ri, ci = ski_draw.polygon(r, c)
    #if mode == None:
    #    mode = choose(['per_channel', 'all_same'])
    #size = 1
    #if mode == 'per_channel':
    #    size = img.shape[2]
#######
##  #TODO
#######
    #img_max = np.max(img)
    #img_min = np.min(img)
    #gap = img_max - img_min
    #if gap <= max_shadow:
    #    if img_max < 2*high:
    #        neg = 1
    #    elif img_min > 255-2*high:
    #        neg = -1
    #if neg is None:
    #    neg = choose([-1,1])
    #weight = neg * rnd.randint(low=low, high=high, size=size, dtype=np.int32)
    #img_copy = np.int32(img)
    #img_copy[ri,ci,:] += weight
    #return np.uint8(np.clip(img_copy, 0, 255))

def drop(img, drop_percent=0.3):
    pass

def project(img, x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, mode=None):
     """
     `project` is not completed
     """
     xs = choose_int(*DEFAULT_X_RANGE, size=4)
     ys = choose_int(*DEFAULT_Y_RANGE, size=4)
     move = np.array(list(zip(ys, xs)))
     oy, ox = img.shape[0]-1, img.shape[1]-1
     orig = np.array(((0,0), (0,ox), (oy,ox), (oy,0)))
     assert(orig.shape == move.shape)
     proj = orig + move
     proj_transform = ski_trs.ProjectiveTransform()
     if mode == None:
         mode = choose(DEFAULT_MODES)
     if proj_transform.estimate(orig, proj):
         #log("Image wasn't projected")
         return img
     return ski_trs.warp(img,
                         proj_transform,
                         output_shape=img.shape,
                         mode=mode,
                         preserve_range=True)

def scale_intensity(img,
             in_range=DEFAULT_INTENSITY_RANGE,
             out_range=None,
             min_gap=INTENCITY_DISTANCE):
    #log("Exposure")
    low, high = DEFAULT_INTENSITY_RANGE
    ogap = np.max(img) - np.min(img)
    gap = np.min([min_gap, ogap])
    if out_range is not None:
        return ski_exs.rescale_intensity(img, out_range=out_range)
    outl = choose_int(low, high - gap)
    outr = choose_int(outl + gap, high)
    out_range = (outl, outr)
    #log(np.min(img), np.max(img), np.max(img) - np.min(img))
    #log(out_range)
    return ski_exs.rescale_intensity(img, in_range=in_range, out_range=out_range)

def noise(img, mode=None):
    #log("Noise")
    if mode == None:
        mode = choose(DEFAULT_NOIZE_MODES)
    assert(mode in DEFAULT_NOIZE_MODES)
    #log(mode)
    with_noise = ski_util.random_noise(img, mode=mode, clip=True, var=0.001)
    return ski_util.img_as_ubyte(with_noise)

#DEFAULT_FUNS = [fliplr, scale_intensity, noise, shadow, scale, translate, rotate]
DEFAULT_FUNS = [fliplr, scale_intensity, noise, scale, translate, rotate]
DEFAULT_STR_FUNS = list(map(lambda fun: fun.__name__, DEFAULT_FUNS))

class ImageRich:
     def __init__(self, include=None, exclude=None):
         self._funs = []
         locs = globals()
         if include == None:
             self._funs = DEFAULT_FUNS
         else:
             self._funs = self.__include(include=include)
         if exclude != None:
             self._funs = self.__exclude(exclude=exclude)
         if self._funs == []:
             raise ValueError("There is no augmentation methods")
     def __exclude(self, exclude=[]):
         locs = globals()
         funs = self._funs
         exclude = np.unique(exclude)
         for m in exclude:
             if m not in locs:
                 raise ValueError("Method unavailable")
             funs.remove(locs[m])
         return funs
     def __include(self, include=[]):
         locs = globals()
         funs = [None] * len(include)
         for i, m in enumerate(include):
             if m not in locs:
                 raise ValueError("Method unavailable")
             funs[i] = locs[m]
         return list(set(self._funs).union(funs))
     def list_methods():
         return DEFAULT_STR_FUNS
     def augment(self, img, include=None, exclude=None, random_funs=True):
         funs = self._funs
         if random_funs:
             funs = choose(funs, random_size=True)
         out = img
         for fun in funs:
             out = fun(out)
         return out
