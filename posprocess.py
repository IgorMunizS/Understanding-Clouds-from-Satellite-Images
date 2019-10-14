import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from utils.utils import make_mask,mask2rle, np_resize
from utils.posprocess import post_process_minsize,draw_convex_hull, post_process
import sys
import argparse


def posprocess(file,mode):
    sub = pd.read_csv(file)
    name = file.split('/')[-1].split('.')[0]

    mode = 'convex'  # choose from 'rect', 'min', 'convex' and 'approx'
    model_class_names = ['Fish', 'Flower', 'Gravel', 'Sugar']
    min_size = [20000, 20000, 22500, 10000]

    img_label_list = []
    enc_pixels_list = []
    test_imgs = os.listdir('../../dados/test_images/')
    for test_img_i, test_img in enumerate(tqdm(test_imgs)):
        for class_i, class_name in enumerate(model_class_names):

            path = os.path.join('../../dados/test_images/', test_img)
            img = cv2.imread(path).astype(np.float32)  # use already-resized ryches' dataset
            img = img / 255.
            img = np_resize(img, (350, 525))
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_label_list.append(f'{test_img}_{class_name}')

            mask = make_mask(sub, test_img + '_' + class_name, shape=(350, 525))
            if True:
                # if class_name == 'Flower' or class_name =='Sugar': # you can decide to post-process for some certain classes
                mask = draw_convex_hull(mask.astype(np.uint8), mode=mode)
            mask[img2 <= 2 / 255.] = 0
            mask = post_process(mask, min_size[class_i])

            if mask.sum() == 0:
                enc_pixels_list.append(np.nan)
            else:
                mask = np.where(mask > 0.5, 1.0, 0.0)
                enc_pixels_list.append(mask2rle(mask))

    name = name + '_convex.csv'
    submission_df = pd.DataFrame({'Image_Label': img_label_list, 'EncodedPixels': enc_pixels_list})
    submission_df.to_csv(name, index=None)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--file', help='Submission file', default=None)
    parser.add_argument('--mode', help='Model backbone', default='convex', type=str)
    parser.add_argument('--shape', help='Shape of resized images', default=(350,525), type=tuple)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)


    posprocess(args.file,args.mode)