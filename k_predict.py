from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce
from keras.optimizers import Adam, Nadam
from utils.preprocess import get_test_data, get_data_preprocessed
from utils.generator import DataGenerator
from utils.utils import build_rles, build_masks, np_resize
from utils.posprocess import post_process
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import cv2
import argparse
import sys
import gc
from tta_wrapper import tta_segmentation
import pickle
import os
from tqdm import tqdm
import segmentation_models as sm


def make_prediction(smmodel, backbone, reshape, n_splits, tta, swa, what_to_make='test'):
    h, w = reshape

    sub_df, test_imgs = get_test_data()
    nb_test_img = test_imgs.shape[0]

    train_df, mask_count_df = get_data_preprocessed()
    print('Len mask_count_df', len(mask_count_df))
    print(mask_count_df.head())
    skf = StratifiedKFold(n_splits=n_splits, random_state=133)

    nb_img_preds = nb_test_img if what_to_make == 'test' else len(mask_count_df)

    all_preds = np.zeros((nb_img_preds, 350, 525, 4), dtype=np.int16)
    cnt, val_names = 0, []
    model = get_model(smmodel, backbone, Adam(), dice_coef_loss_bce, dice_coef, reshape)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):
        print('Predicting fold', n_fold, '...')

        model_fold_filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold)

        if swa:
            model_fold_filepath = model_fold_filepath + '_swa.h5'
        else:
            model_fold_filepath = model_fold_filepath + '.h5'


        model.load_weights(model_fold_filepath)

        if tta: model = tta_segmentation(model, h_flip=True, h_shift=(-15, 15),
                                         v_flip=True, v_shift=(-15, 15), contrast=(-0.9, 0.9), merge='gmean')

        if what_to_make == 'test':
            for i, img_name in tqdm(enumerate(test_imgs.ImageId)):
                single_img = load_img(img_name, mode='test', backbone=backbone, reshape=reshape)
                Y = model.predict(single_img)
                Y = reshape_to_submission_single(Y[0])
                all_preds[i] += Y

        elif what_to_make == 'valid':
            for i, img_name in tqdm(enumerate(mask_count_df.iloc[val_indices]['ImageId'])):
                single_img = load_img(img_name, mode='valid', backbone=backbone, reshape=reshape)
                Y = model.predict(single_img)
                Y = reshape_to_submission_single(Y[0])
                all_preds[cnt] = Y
                val_names.append(img_name)
                cnt += 1

    if what_to_make == 'test':
        all_preds = (all_preds // 6).astype(np.int8)
        filesave = '../predictions/' + str(smmodel) + '_' + str(backbone) + '_' + str(n_splits) + '_test_avg_tta_more.npy'

    elif what_to_make == 'valid':
        all_preds = all_preds.astype(np.int8)
        filesave = '../predictions/' + str(smmodel) + '_' + str(backbone) + '_' + str(n_splits) + '_oof_tta.npy'
        # np.save('../predictions/' + str(smmodel) + '_' + str(backbone) + '_' + str(n_splits) + '_oof_imgnames_tta.npy', val_names)

    np.save(filesave, all_preds)
    print('Prediction file saved to', filesave)


def load_img(img_name, mode='test', backbone='efficientnetb3', reshape=(320, 480)):
    if mode == 'test':
        path = '../../dados/test_images/'
    else:
        path = '../../dados/train_images/'

    _preprocess = sm.get_preprocessing(backbone)

    img_path = f"{path}/{img_name}"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np_resize(img, reshape)
    img = np.expand_dims(img, axis=0)
    img = _preprocess(img)
    return img


def reshape_to_submission_single(pred):
    output = np.zeros((350, 525, 4), dtype=np.int16)
    for t in range(4):
        pred_layer = pred[:, :, t]
        pred_layer = cv2.resize(pred_layer, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        pred_layer = (pred_layer*100).astype(np.int16)
        output[:, :, t] = pred_layer
    return output


def reshape_to_submission(batch_pred):
    # Input batch_pred shape: batch_size x 320 x 480 x 4
    len_batch = batch_pred.shape[0]
    output = np.zeros((len_batch, 350, 525, 4), dtype=np.int16)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    for j, pred in enumerate(batch_pred):  # pred: 320 x 480 x 4
        for t in range(4):
            pred_layer = pred[:, :, t]
            pred_layer = cv2.resize(pred_layer, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)  # pred_mask
            pred_layer = (sigmoid(pred_layer) * 100).astype(np.int16)

            output[j, :, :, t] = pred_layer

    return output


def parse_args(args):
    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='efficientnetb3', type=str)
    parser.add_argument('--shape', help='Shape of resized images', nargs='+', default=(320, 480), type=int)
    parser.add_argument('--tta', help='Shape of resized images', default=False, type=bool)
    parser.add_argument('--swa', help='swa or not', default=False, type=bool)
    parser.add_argument('--n_splits', help='n_fold', default=6, type=int)
    parser.add_argument('--what_to_make', help='test or valid', default='test', type=str)
    parser.add_argument("--cpu", default=False, type=bool)


    return parser.parse_args(args)


if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    make_prediction(smmodel=args.model, backbone=args.backbone, reshape=args.shape,
                    n_splits=args.n_splits, tta=args.tta, swa=args.swa, what_to_make=args.what_to_make)