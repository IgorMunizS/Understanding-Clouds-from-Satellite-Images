import argparse
import sys
from sklearn.model_selection import StratifiedKFold
from utils.preprocess import get_data_preprocessed
from sklearn.metrics import f1_score
from utils.generator import DataGenerator
from keras_radam import RAdam
from keras.optimizers import Adam, Nadam, SGD
from utils.lr import CyclicLR, Lookahead, AdamAccumulate
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce, lovasz_loss, combo_loss, np_dice_coef,dice
from utils.callbacks import ValPosprocess
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
from imblearn.over_sampling import RandomOverSampler
import itertools
from predict import predict_postprocess
from utils.posprocess import post_process
import ray
import psutil
import numpy as np
from utils.posprocess import draw_convex_hull, post_process_minsize
from utils.utils import mask2rle
import os
import pandas as pd
import time

@ray.remote
def parallel_post_process(y_true,y_pred,class_id,t,ms,shape):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    masks = []
    for i in range(y_pred.shape[0]):
        probability = y_pred[i, :, :, class_id]
        predict, num_predict = post_process(sigmoid(probability), t, ms, shape)
        masks.append(predict)

    d = []
    for i, j in zip(masks, y_true[:, :, :, class_id]):
        if (i.sum() == 0) & (j.sum() == 0):
            d.append(1)
        else:
            d.append(np_dice_coef(i, j))

    return d

def evaluate(smmodel,backbone,nfold,shape=(320,480)):

    # if shape is None:
    #     shape = (1400,2100)


    train_df, mask_count_df = get_data_preprocessed()
    opt = Nadam(lr=0.0002)
    model = get_model(smmodel, backbone, opt, dice_coef_loss_bce, dice_coef, shape)

    skf = StratifiedKFold(n_splits=5, random_state=133)
    oof_data = []
    oof_predicted_data =[]
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=4)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):


        if n_fold >= nfold:
            print('Evaluating fold number ',str(n_fold))


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

            _ ,y_true = val_generator.__getitem__(0)
            val_generator.batch_size = 1

            filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '.h5'
            model.load_weights(filepath)

            # results = model.evaluate_generator(
            #     val_generator,
            #     workers=40,
            #     verbose=1
            # )
            # print(results)

            y_pred = model.predict_generator(
                val_generator,
                workers=40,
                verbose=1
            )
            print(y_true.shape)
            print(y_pred.shape)
            # print(y_pred)
            print("Dice: ", np_dice_coef(y_true, y_pred))

            oof_data.extend(y_true.astype(np.float16))
            oof_predicted_data.extend(y_pred.astype(np.float16))
            del y_true, y_pred
            gc.collect()

    del val_generator, model
    gc.collect()

    oof_data = np.array(oof_data)
    oof_predicted_data = np.array(oof_predicted_data)
    print(oof_data.shape)
    print(oof_predicted_data.shape)
    print("CV Final Dice: ", np_dice_coef(oof_data, oof_predicted_data))
    oof_data = ray.put(oof_data.astype(np.float32))
    oof_predicted_data = ray.put(oof_predicted_data.astype(np.float32))

    now = time.time()
    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(40, 80, 5):
            t /= 100
            for ms in range(10000, 30000, 1000):

                d = ray.get([parallel_post_process.remote(oof_data,oof_predicted_data,class_id,t,ms,shape)])

                # print(t, ms, np.mean(d))
                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        print('Time: ', time.time() - now)
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        class_params[class_id] = (best_threshold, best_size)
        ray.shutdown()
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
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='resnet34', type=str)
    parser.add_argument('--shape', help='Shape of resized images', default=(320, 480), type=tuple)
    parser.add_argument('--nfold', help='number of fold to evaluate', default=0, type=int)
    parser.add_argument("--cpu", default=False, type=bool)



    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    evaluate(args.model,args.backbone,args.nfold,args.shape)