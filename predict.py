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
import gc
from tta_wrapper import tta_segmentation
import pickle

def predict(batch_idx,test_imgs,shape,sub_df,backbone,TTA,model):
    h,w = shape
    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=shape,
        n_channels=3,
        base_path='../../dados/test_images/',
        target_df=sub_df,
        batch_size=43,
        n_classes=4,
        backbone=backbone
    )

    if TTA:
        test_generator.batch_size = 1
        tta_model = tta_segmentation(model, h_flip=True,h_shift=(-15, 15),
                                     input_shape=(h, w, 3), merge='mean')

        batch_pred_masks = tta_model.predict_generator(
            test_generator,
            workers=40,
            verbose=1
        )
    else:
        batch_pred_masks = model.predict_generator(
            test_generator,
            workers=40,
            verbose=1
        )

    return batch_pred_masks

def predict_postprocess(batch_idx,posprocess,batch_pred_masks,shape=(350,525),minsize=None,threshold=0.6):
    if minsize is None:
        minsizes = [20000, 20000, 22500, 10000]
    else:
        minsizes = minsize

    h,w = shape
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    all_masks =[]

    for j, b in enumerate(batch_idx):

        if posprocess:
            pred_masks = batch_pred_masks[j,]
            pred_masks = cv2.resize(pred_masks, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            arrt = np.array([])
            for t in range(4):
                a, num_predict = post_process(sigmoid(pred_masks[:, :, t]),threshold, minsizes[t], shape)

                if (arrt.shape == (0,)):
                    arrt = a.reshape(h, w, 1)
                else:
                    arrt = np.append(arrt, a.reshape(h, w, 1), axis=2)

            pred_masks = arrt
        else:
            pred_masks = batch_pred_masks[j,].round().astype(int)

        all_masks.append(pred_masks)

    return all_masks

def convert_masks_for_submission(batch_idx,test_imgs,sub_df,prediction):
    test_df = []
    for j, b in enumerate(batch_idx):
        filename = test_imgs['ImageId'].iloc[b]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        pred_rles = build_rles(prediction[j,], reshape=(350, 525))

        image_df['EncodedPixels'] = pred_rles
        test_df.append(image_df)


    return test_df



def predict_fold(fold_number,smmodel, backbone,model,batch_idx,test_imgs,shape,sub_df,TTA):

    print('Predicting Fold ', str(fold_number))
    filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(fold_number) + '_swa.h5'
    model.load_weights(filepath)

    batch_pred_masks = predict(batch_idx, test_imgs, shape, sub_df, backbone, TTA, model)

    return batch_pred_masks

def save_prediction(prediction, name):
    with open('../predictions/' + name +'_.pickle', 'wb') as handle:
        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)


def final_predict(models,folds,shape,TTA=False,posprocess=False):

    sub_df,test_imgs = get_test_data()
    print(test_imgs.shape[0])
    # batch_idx = list(range(test_imgs.shape[0]))
    test_df = []
    batch_pred_emsemble=[]
    submission_name = ''

    for smmodel,backbone in models:
        print('Predicting {} {}'.format(smmodel,backbone))
        opt = Adam()
        model = get_model(smmodel, backbone, opt, dice_coef_loss_bce, dice_coef, shape)
        model_masks=[]
        submission_name = submission_name + str(smmodel) + '_' + str(backbone) + '_'

        for i in range(0, test_imgs.shape[0], 860):
            batch_idx = list(
                range(i, min(test_imgs.shape[0], i + 860))
            )
            fold_result = []

            for i in folds:

                batch_pred_masks = predict_fold(i,smmodel, backbone,model,batch_idx,test_imgs,shape,sub_df,TTA)
                # print(np.array(batch_pred_masks).shape)
                fold_result.append(batch_pred_masks)

            batch_pred_masks = np.mean(fold_result, axis=0)
            model_masks.extend(batch_pred_masks)

        batch_pred_emsemble.append(model_masks)

        del model, model_masks,batch_pred_masks,fold_result
        gc.collect()


    pred_emsemble = np.mean(batch_pred_emsemble, axis=0)

    save_prediction(pred_emsemble, submission_name)

    batch_idx = list(range(test_imgs.shape[0]))
    # masks_posprocessed = predict_postprocess(batch_idx,test_imgs,sub_df,posprocess,batch_pred_emsemble)
    pred_emsemble = np.array(predict_postprocess(batch_idx, posprocess, pred_emsemble))

    print(pred_emsemble.shape)
    test_df = convert_masks_for_submission(batch_idx,test_imgs,sub_df,pred_emsemble)
    submission_name = submission_name + '.csv'
    generate_submission(test_df, submission_name)


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
    parser.add_argument('--shape', help='Shape of resized images', default=(384, 576), type=tuple)
    parser.add_argument('--tta', help='Shape of resized images', default=False, type=bool)
    parser.add_argument('--posprocess', help='Shape of resized images', default=False, type=bool)
    parser.add_argument('--fold', help='Fold number to predict', default=None, nargs='+', type=int)
    parser.add_argument('--emsemble', help='Do model emsemble', default=False, type=bool)


    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.fold is None:
        folds = [0,1,2,3,4]
    else:
        folds = args.fold


    if args.emsemble:
        models = [['unet','efficientnetb4'],['unet','efficientnetb3']]
    else:
        models = [[args.model, args.backbone]]


    final_predict(models,folds,args.shape,args.tta,args.posprocess)