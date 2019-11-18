from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce
from keras.optimizers import Adam, Nadam
from utils.preprocess import get_test_data
from utils.generator import DataGenerator
from utils.utils import build_rles, np_resize
from utils.posprocess import post_process
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
        tta_model = tta_segmentation(model, h_flip=True, v_flip=True,
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

def predict_postprocess(batch_idx,posprocess,batch_pred_masks,shape=(350,525),minsize=None,threshold=None, bottom_threshold=None,fixshape=False):
    if minsize is None:
        minsize = [10000, 10000, 10000, 10000]

    if threshold is None:
        threshold = [0.6, 0.6, 0.6, 0.6]

    if bottom_threshold is None:
        bottom_threshold = [None,None,None,None]

    h,w = (350,525)
    resize = False if shape == (350,525) else True
    # sigmoid = lambda x: 1 / (1 + np.exp(-x))

    all_masks =[]

    for j, b in enumerate(batch_idx):

        if posprocess:
            pred_masks = np.float32(batch_pred_masks[j,])
            # pred_masks = cv2.resize(np.float32(pred_masks), dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            arrt = np.array([])
            for t in range(4):
                a, num_predict = post_process(pred_masks[:, :, t],threshold[t], minsize[t], bottom_threshold[t],shape,fixshape)
                a = cv2.resize(a,dsize=(w, h), interpolation=cv2.INTER_LINEAR)
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



def predict_fold(fold_number,smmodel, backbone,model,batch_idx,test_imgs,shape,sub_df,TTA,swa):

    print('Predicting Fold ', str(fold_number))
    filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(fold_number)
    if swa:
        filepath += '_swa.h5'
    else:
        filepath += '.h5'
    model.load_weights(filepath)

    batch_pred_masks = predict(batch_idx, test_imgs, shape, sub_df, backbone, TTA, model)

    return batch_pred_masks

def predict_multimodel(fold_number,smmodel, backbone,model,batch_idx,test_imgs,shape,sub_df,TTA,swa):
    classes=['Fish','Flower','Gravel','Sugar']
    batch_pred_masks = np.zeros((len(batch_idx), *shape, 4), dtype=np.float32)
    for i,cls in enumerate(classes):
        print('Predicting Fold ', str(fold_number))
        print('Predicting Fold ', str(fold_number))
        filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(fold_number) + '_' + cls
        if swa:
            filepath += '_swa.h5'
        else:
            filepath += '.h5'
        model.load_weights(filepath)

        batch_pred_masks[:,:,:,i] = predict(batch_idx, test_imgs, shape, sub_df, backbone, TTA, model)[:,:,:,0]

    return batch_pred_masks

def save_prediction(prediction, name):
    resize_prediction = np.zeros((prediction.shape[0], 350,525,4), dtype=np.float16)
    for i in range(prediction.shape[0]):
        resize_prediction[i, :, :, :] = np_resize(prediction[i, :, :, :].astype(np.float32), (350, 525)).astype(np.float16)

    # with open('../predictions/' + name +'_.pickle', 'wb') as handle:
    #     pickle.dump(resize_prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save('../predictions/' + name +'_.npy', resize_prediction)

def final_predict(models,folds,shape,TTA=False,posprocess=False,swa=False,minsizes=None,thresholds=None,fixshape=False,
                  multimodel=False):

    sub_df,test_imgs = get_test_data()
    print(test_imgs.shape[0])
    # batch_idx = list(range(test_imgs.shape[0]))
    test_df = []
    batch_pred_emsemble=[]
    submission_name = ''

    for smmodel,backbone in models:
        print('Predicting {} {}'.format(smmodel,backbone))
        opt = Adam()
        model_masks=[]
        submission_name = submission_name + str(smmodel) + '_' + str(backbone) + '_'

        for i in range(0, test_imgs.shape[0], 860):
            batch_idx = list(
                range(i, min(test_imgs.shape[0], i + 860))
            )
            fold_result = []

            for i in folds:
                model = get_model(smmodel, backbone, opt, dice_coef_loss_bce, [dice_coef])

                if multimodel:
                    batch_pred_masks = predict_multimodel(i,smmodel, backbone,model,batch_idx,test_imgs,shape,sub_df,TTA,swa)
                else:
                    batch_pred_masks = predict_fold(i,smmodel, backbone,model,batch_idx,test_imgs,shape,sub_df,TTA,swa)
                # print(np.array(batch_pred_masks).shape)
                fold_result.append(batch_pred_masks.astype(np.float16))

            batch_pred_masks = np.mean(fold_result, axis=0, dtype=np.float16)
            del fold_result
            gc.collect()

            model_masks.extend(batch_pred_masks.astype(np.float16))
            del batch_pred_masks
            gc.collect()

        batch_pred_emsemble.append(model_masks)

        del model, model_masks
        gc.collect()

    batch_pred_emsemble = np.mean(batch_pred_emsemble, axis=0, dtype=np.float16)

    if TTA:
        submission_name += '_tta'

    save_prediction(batch_pred_emsemble, submission_name)
    batch_idx = list(range(test_imgs.shape[0]))
    # print(pred_emsemble.shape)
    batch_pred_emsemble = np.array(predict_postprocess(batch_idx, posprocess, batch_pred_emsemble, shape=shape,
                                                       minsize=minsizes,threshold=thresholds, fixshape=fixshape))

    test_df = convert_masks_for_submission(batch_idx,test_imgs,sub_df,batch_pred_emsemble)
    submission_name = submission_name + '.csv'
    generate_submission(test_df, submission_name)

def postprocess_pickle(pickle_path, emsemble, minsizes, thresholds, bottom_threshold=None,yves=False,fixshape=False):

    sub_df, test_imgs = get_test_data()
    print(test_imgs.shape[0])

    if emsemble:
        models_emsemble_path = pickle_path
        submission_name = 'emsemble_submission'
        for i,file in enumerate(os.listdir(models_emsemble_path)):
            if i == 0:
                predicted_data = np.load(models_emsemble_path + file)
            else:
                predicted_data += np.load(models_emsemble_path + file)

        predicted_data /= len(os.listdir(models_emsemble_path))
    else:
        submission_name = pickle_path.split('/')[-1].split('.')[0]

        try:
            with open(pickle_path, 'rb') as handle:
                predicted_data = pickle.load(handle)
        except:
            predicted_data = np.load(pickle_path)

    if yves:
        for x,img in tqdm(enumerate(predicted_data)):
            for k in [0,1,2,3]:
                im_layer = img[:,:,k]
                max_col = np.max(im_layer, axis=1)
                max_row = np.max(im_layer, axis=0)
                mat = [max_col[i] + max_row[j] for i in range(h) for j in range(w)]
                mat = np.reshape(mat, (h,w))
                im_layer = 0.7*im_layer + 0.3*mat

                predicted_data[x,:,:,k] = im_layer

    batch_idx = list(range(test_imgs.shape[0]))
    # masks_posprocessed = predict_postprocess(batch_idx,test_imgs,sub_df,posprocess,batch_pred_emsemble)
    pred_emsemble = np.array(predict_postprocess(batch_idx, True, predicted_data, minsize=minsizes,threshold=thresholds,
                                                 bottom_threshold=bottom_threshold,fixshape=fixshape))

    print(pred_emsemble.shape)
    test_df = convert_masks_for_submission(batch_idx, test_imgs, sub_df, pred_emsemble)
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
    parser.add_argument('--shape', help='Shape of resized images', nargs='+', default=[320, 480], type=int)
    parser.add_argument('--tta', help='Shape of resized images', default=False, type=bool)
    parser.add_argument('--swa', help='Apply SWA', default=False, type=bool)
    parser.add_argument('--posprocess', help='Shape of resized images', default=False, type=bool)
    parser.add_argument('--fold', help='Fold number to predict', default=None, nargs='+', type=int)
    parser.add_argument('--emsemble', help='Do model emsemble', default=False, type=bool)
    parser.add_argument('--prediction', help='Pickle path for prediction', default=None, type=str)
    parser.add_argument('--minsizes', nargs='+', default=None, type=int)
    parser.add_argument('--thresholds', nargs='+', default=None, type=float)
    parser.add_argument('--bottom', nargs='+', default=None, type=float)
    parser.add_argument('--fixshape', default=False, type=bool)
    parser.add_argument('--yves', default=False, type=bool)
    parser.add_argument('--multimodel', default=False, type=bool)

    parser.add_argument("--cpu", default=False, type=bool)


    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.fold is None:
        folds = [0,1,2,3,4,5]
    else:
        folds = args.fold

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    if args.emsemble:
        models = [['unet','efficientnetb4'],['unet','efficientnetb3']]
    else:
        models = [[args.model, args.backbone]]



    h,w = args.shape
    if args.prediction is not None:
        postprocess_pickle(args.prediction, args.emsemble, args.minsizes,args.thresholds, args.bottom,args.yves, args.fixshape)
    else:
        final_predict(models,folds,(h,w),args.tta,args.posprocess,args.swa,args.minsizes,args.thresholds,
                      args.fixshape,args.multimodel)