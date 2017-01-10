import skimage.transform as ski_trs
import skimage.exposure as ski_exs
import skimage.util as ski_util
import skimage.draw as ski_draw
import numpy as np
import cv2 as cv
import numpy.random as rnd

IMAGE_Y_LEN = 100
IMAGE_X_LEN = 100
IMAGE_SHAPE = (IMAGE_Y_LEN, IMAGE_X_LEN)
DEFAULT_SHIFT_RANGE = ((-10,10), (-10,10))
DEFAULT_ANGLE_RANGE = (-45,45)
DEFAULT_SCALE_RANGE = (0.7, 1.3)
INTENCITY_DISTANCE = 95
DEFAULT_MODES = ['constant', 'edge', 'symmetric', 'reflect', 'wrap']
DEFAULT_NOIZE_MODES = ['gaussian', 'localvar', 'poisson',
                       'salt', 'pepper', 's&p', 'speckle']
ORIGIN = (0, 0)

def flip_coin():
    return bool(random.randint(0, 2))

def choose(choices, random_size=False, replace=False):
    if random_size:
        size = choose_int(1, len(choices))
        return rnd.choice(choices, size=size, replace=replace)
    return rnd.choice(choices)

def choose_mode():
    return rnd.choice(DEFAULT_MODES)

def choose_int(*args):
    assert(len(args) == 2)
    return rnd.randint(low=args[0], high=args[1]+1)

def clip(a, a_min=0, a_max=255):
    return np.clip(a, a_min, a_max)

def choose_float(*args):
    assert(len(args) == 2)
    return rnd.uniform(low=args[0], high=args[1])

def fliplr(img):
    #print("Fliplr")
    return np.fliplr(img)

def rotate(img, range_angles=DEFAULT_ANGLE_RANGE, mode=None):
    #print("Rotate")
    assert(len(range_angles) == 2)
    angle = choose_int(*range_angles)
    if mode is None:
        mode = choose_mode()
    assert(mode in DEFAULT_MODES)
    #print("Angle=", angle)
    rotated = ski_trs.rotate(img, angle=angle, mode=mode, preserve_range=True)
    return np.uint8(rotated)

def translate(img, range_shift=DEFAULT_SHIFT_RANGE, mode=None):
    #print("Translate")
    dx = choose_int(*range_shift[1])
    dy = choose_int(*range_shift[0])
    affine = ski_trs.AffineTransform(translation=(dx, dy))
    if mode is None:
        mode = choose_mode()
    assert(mode in DEFAULT_MODES)
    translated = ski_trs.warp(img, affine, mode=mode, preserve_range=True)
    return np.uint8(translated)

def scale(img, range_factor=DEFAULT_SCALE_RANGE, mode=None):
    #print("Scale")
    assert(len(range_factor) == 2)
    factor = choose_float(*range_factor)
    if factor == 1.0:
        return img
    scaled_img = ski_trs.rescale(img, scale=factor, preserve_range=True)
    scaled_img = np.uint8(scaled_img)
    shape = img.shape
    scaled_shape = scaled_img.shape
    if factor < 1:
        if mode is None:
            mode = choose_mode()
        assert(mode in DEFAULT_MODES)
        pad_fun = (lambda i: ((shape[i] - scaled_shape[i]) // 2,
                              (shape[i] - scaled_shape[i]) // 2 + scaled_shape[i] % 2))
        pad = (pad_fun(0), pad_fun(1), ORIGIN)
        scaled_img = ski_util.pad(scaled_img, mode=mode, pad_width=pad)
    else:
        range_y = (0, scaled_shape[0] - shape[0])
        wy = choose_int(*range_y)
        after_wy = scaled_shape[0] - wy - shape[0] 
        range_x = (0, scaled_shape[1] - shape[1])
        wx = choose_int(*range_x)
        after_wx = scaled_shape[1] - wx - shape[1]
        crop = ((wy, after_wy), (wx, after_wx), ORIGIN)
        scaled_img = ski_util.crop(scaled_img, crop_width=crop)
    assert(scaled_img.shape == img.shape)
    return scaled_img

def shadow(img, mode=None, low=10, high=100):
    shape = img.shape
    sy, sx = shape[0], shape[1]
    choose_side = lambda: choose([0, sx-1])
    rand_x = lambda: np.clip(rnd.normal(loc=sx//2, scale=sx//4), 0, sx-1)
    xo, xu, xb = choose_side(), rand_x(), rand_x()
    yu, yb = 0, sy-1
    r = np.array([yu, yu, yb, yb, yu])
    c = np.array([xo, xu, xb, xo, xo])
    ri, ci = ski_draw.polygon(r, c)
    if mode == None:
        mode = choose(['per_channel', 'all_same'])
    size = 1
    if mode == 'per_channel':
        size = img.shape[2]
    neg = choose([-1,1])
    weight = neg * rnd.randint(low=low, high=high, size=size, dtype=np.int32)
    img_copy = np.int32(img)
    img_copy[ri,ci,:] += weight
    return np.uint8(np.clip(img_copy, 0, 255))

def rowcol_drop(img, drop_percent=0.3):
    pass

def exposure(img, in_range=None, out_range=None, min_gap=INTENCITY_DISTANCE):
    low, high, gap = 0, 255, min_gap
    standard = (low, high)
    assert(low < high)
    assert(gap < high)
    assert(gap > low)
    exposure_types = ['in', 'out', 'both']
    choice = choose(exposure_types)
    if in_range is None:
        if choice == 'out':
            in_range = standard
        else:
            inl = choose_int(low, high - gap)
            inr = choose_int(inl + gap, high)
            in_range = (inl, inr)
    if out_range is None:
        if choice == 'in':
            out_range = standard
        else:
            outl = choose_int(low, high - gap)
            outr = choose_int(outl + gap, high)
            out_range = (outl, outr)
    return ski_exs.rescale_intensity(img, in_range=in_range, out_range=out_range)

def noise(img, mode=None):
    if mode == None:
        mode = choose(DEFAULT_NOIZE_MODES)
    assert(mode in DEFAULT_NOIZE_MODES)
    with_noise = ski_util.random_noise(img, mode=mode)
    return ski_util.img_as_ubyte(with_noise)

class ImageRich:
     def __init__(self, seed=0, methods=None):
         rnd.seed(seed)
         funs = []
         if methods == None:
             funs = [fliplr, exposure, noise, shadow, scale, translate, rotate]
         else:
             funs = [None] * len(methods)
             locs = locals()
             for i, m in enumerate(methods):
                 if m not in locs:
                     raise ValueError("Method unavailable")
                 funs[i] = locs[m]
         self._funs = funs
     def exec(self, img, random_funs=True):
         funs = self._funs
         if random_funs:
             funs = choose(funs, random_size=True)
         out = img
         for fun in funs:
             out = fun(out)
         return out
