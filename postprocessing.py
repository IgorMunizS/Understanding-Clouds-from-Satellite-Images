import cv2
from utils.utils import make_mask,mask2rle, np_resize
import argparse
import sys
from sklearn.model_selection import StratifiedKFold
from utils.preprocess import get_data_preprocessed
from utils.generator import DataGenerator
from keras.optimizers import Adam, Nadam, SGD
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce, lovasz_loss, combo_loss, np_dice_coef, dice
import gc
from utils.posprocess import post_process
import ray
import psutil
import numpy as np
from utils.posprocess import draw_convex_hull, post_process_minsize
from utils.utils import mask2rle
import os
import pandas as pd
import time
from tqdm import tqdm


def postprocess_shape(file,mode):
    sub = pd.read_csv(file)
    name = file.split('/')[-1].split('.')[0]

    # mode = 'convex'  # choose from 'rect', 'min', 'convex' and 'approx'
    model_class_names = ['Fish', 'Flower', 'Gravel', 'Sugar']
    min_size = [25000, 15000, 22500, 10000]

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
            mask = post_process_minsize(mask, min_size[class_i])

            if mask.sum() == 0:
                enc_pixels_list.append(np.nan)
            else:
                mask = np.where(mask > 0.5, 1.0, 0.0)
                enc_pixels_list.append(mask2rle(mask))

    name = name + '_convex.csv'
    submission_df = pd.DataFrame({'Image_Label': img_label_list, 'EncodedPixels': enc_pixels_list})
    submission_df.to_csv(name, index=None)


# @ray.remote
def parallel_post_process(y_true, y_pred, class_id, t, ms, shape):
    # sigmoid = lambda x: 1 / (1 + np.exp(-x))

    masks = []
    for i in range(y_pred.shape[0]):
        probability = y_pred[i, :, :, class_id].astype(np.float32)
        predict, num_predict = post_process(probability, t, ms, shape)
        masks.append(predict)

    d = []
    for i, j in zip(masks, y_true[:, :, :, class_id]):
        i = i.astype(np.float32)
        j = j.astype(np.float32)
        if (i.sum() == 0) & (j.sum() == 0):
            d.append(1)
        else:
            d.append(np_dice_coef(i, j))

    return d

def evaluate(smmodel, backbone, nfold, shape=(320, 480)):

    n_splits = 6
    # if shape is None:
    #     shape = (1400,2100)

    train_df, mask_count_df = get_data_preprocessed()

    skf = StratifiedKFold(n_splits=n_splits, random_state=133)
    oof_data = []
    oof_predicted_data = []
    # num_cpus = psutil.cpu_count(logical=False)
    # ray.init(num_cpus=4)
    oof_dice = []

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):

        if n_fold >= nfold:
            print('Evaluating fold number ', str(n_fold))

            val_generator = DataGenerator(
                val_indices,
                df=mask_count_df,
                shuffle=False,
                target_df=train_df,
                batch_size=len(val_indices),
                reshape=shape,
                augment=False,
                n_channels=3,
                n_classes=4,
                backbone=backbone
            )

            _, y_true = val_generator.__getitem__(0)

            oof_data.extend(y_true.astype(np.float16))

            del y_true
            gc.collect()

    del val_generator
    gc.collect()

    oof_data = np.asarray(oof_data)
    oof_predicted_data = np.load('../predictions/' + str(smmodel) + '_' + str(backbone) + '_' + str(n_splits) + '_oof_tta.npy')
    print(oof_data.shape)
    print(oof_predicted_data.shape)
    print("CV Final Dice: ", np.mean(oof_dice))

    now = time.time()
    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in tqdm(range(55, 80, 5)):
            t /= 100
            for ms in range(10000, 31000, 5000):
                d = parallel_post_process(oof_data, oof_predicted_data, class_id, t, ms, shape)

                # print(t, ms, np.mean(d))
                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        print('Time: ', time.time() - now)
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        class_params[class_id] = (best_threshold, best_size)
        # ray.shutdown()
        # shape_posprocess_list = ['rect', 'min', 'convex', 'approx']

        # for mode in shape_posprocess_list:
        #     pred_masks=[]
        #     for mask in y_pred:
        #         class_masks=np.zeros((mask.shape[0], mask.shape[1], 4))
        #         for i in range(4):
        #             class_pred_masks = np.array(draw_convex_hull((mask[:,:,i]*255).astype(np.uint8), mode))
        #             # class_pred_masks = post_process_minsize(class_pred_masks, 10000)
        #             class_masks[:,:,i] = class_pred_masks
        #         pred_masks.append(class_masks)
        #     print(mode)
        #     print("Dice with shape process: ", np_dice_coef(y_true, np.array(pred_masks)))

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Post Processing Script')
    parser.add_argument('--file', help='predict file path', default=None)
    parser.add_argument('--mode', help='Segmentation model', default='convex')

    parser.add_argument('--prediction', help='predict file path', default=None)
    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='resnet34', type=str)
    parser.add_argument('--shape', help='Shape of resized images', default=(350, 525), type=tuple)
    parser.add_argument('--nfold', help='number of fold to evaluate', default=0, type=int)
    parser.add_argument("--cpu", default=False, type=bool)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    postprocess_shape(args.file,args.mode)


