from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce
from keras.optimizers import Adam, Nadam
from utils.preprocess import get_test_data
from utils.generator import DataGenerator
from utils.utils import build_rles
from utils.posprocess import post_process
import numpy as np
import pandas as pd
import cv2
import argparse
import sys

def predict_fold(smmodel,backbone,shape,TTA=False,posprocess=False):

    opt = Adam()
    model = get_model(smmodel, backbone, opt, dice_coef_loss_bce, dice_coef)
    sub_df,test_imgs = get_test_data()
    batch_idx = list(range(test_imgs.shape[0]))
    fold_result=[]
    test_df = []

    for i in range(5):
        filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(i) + '.h5'
        model.load_weights(filepath)
        test_df = []


        test_generator = DataGenerator(
            batch_idx,
            df=test_imgs,
            shuffle=False,
            mode='predict',
            dim=(350, 525),
            reshape=shape,
            n_channels=3,
            base_path='../dados/test_images/',
            target_df=sub_df,
            batch_size=32,
            n_classes=4,
            backbone=backbone
        )

        batch_pred_masks = model.predict_generator(
            test_generator,
            workers=40,
            verbose=1
        )

        if TTA:
            tta_results = []
            tta_results.append(batch_pred_masks)
            test_generator = DataGenerator(
                batch_idx,
                df=test_imgs,
                shuffle=False,
                mode='predict',
                dim=(350, 525),
                reshape=shape,
                n_channels=3,
                base_path='../dados/test_images/',
                target_df=sub_df,
                batch_size=32,
                n_classes=4,
                backbone=backbone,
                TTA=TTA
            )

            batch_pred_masks = model.predict_generator(
                test_generator,
                workers=40,
                verbose=1
            )
            batch_pred_masks = test_generator.do_tta(batch_pred_masks) #undo TTA (Horizontal and Vertical Flip)
            tta_results.append(batch_pred_masks)

            batch_pred_masks = np.mean(batch_pred_masks,axis=0)

        fold_result.append(batch_pred_masks)


    batch_pred_masks = np.mean(fold_result, axis=0)
    minsizes = [20000, 20000, 22500, 10000]

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    for j, b in enumerate(batch_idx):
        filename = test_imgs['ImageId'].iloc[b]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        pred_masks = batch_pred_masks[j, ].round().astype(int)

        if posprocess:
            pred_masks = cv2.resize(pred_masks, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            arrt = np.array([])
            for t in range(4):
                a, num_predict = post_process(sigmoid(pred_masks[:, :, t]), 0.4, minsizes[t])

                if (arrt.shape == (0,)):
                    arrt = a.reshape(350, 525, 1)
                else:
                    arrt = np.append(arrt, a.reshape(350, 525, 1), axis=2)

            pred_masks = arrt

        pred_rles = build_rles(pred_masks, reshape=(350, 525))

        image_df['EncodedPixels'] = pred_rles
        test_df.append(image_df)

    submission_name = str(smmodel) + '_' + str(backbone) + '.csv'
    generate_submission(submission_name)

def generate_submission(test_df, name):

    test_df = pd.concat(test_df)
    test_df.drop(columns='ImageId', inplace=True)
    test_df.to_csv('../submissions/' + name, index=False)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='resnet34', type=str)
    parser.add_argument('--shape', help='Shape of resized images', default=(320, 480), type=tuple)
    parser.add_argument('--tta', help='Shape of resized images', default=True, type=bool)
    parser.add_argument('--posprocess', help='Shape of resized images', default=True, type=bool)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)


    predict_fold(args.model,args.backbone,args.shape,args.tta,args.posprocess)