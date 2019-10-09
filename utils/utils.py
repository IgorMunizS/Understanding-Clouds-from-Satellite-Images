import numpy as np
import cv2

def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width),
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv2.resize(img, (width, height))


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T


def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))

    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask

    return masks


def build_rles(masks, reshape=None):
    width, height, depth = masks.shape

    rles = []

    for i in range(depth):
        mask = masks[:, :, i]

        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)

        rle = mask2rle(mask)
        rles.append(rle)

    return rles


def rle_decode(mask_rle: str = '', shape=(1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


def make_mask(df, image_label, shape=(1400, 2100), cv_shape=(525, 350), debug=False):
    """
    Create mask based on df, image name and shape.
    """
    if debug:
        print(shape, cv_shape)
    df = df.set_index('Image_Label')
    encoded_mask = df.loc[image_label, 'EncodedPixels']
    #     print('encode: ',encoded_mask[:10])
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded_mask is not np.nan:
        mask = rle_decode(encoded_mask, shape=shape)  # original size

    return cv2.resize(mask, cv_shape)