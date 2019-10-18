import argparse
import sys
from sklearn.model_selection import ShuffleSplit
from utils.preprocess import get_data_preprocessed
from sklearn.metrics import f1_score
from utils.generator import DataGenerator
from keras_radam import RAdam
from keras.optimizers import Adam, Nadam, SGD
from utils.lr import CyclicLR, Lookahead, AdamAccumulate
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce, lovasz_loss, combo_loss, np_dice_coef
from utils.callbacks import ValPosprocess
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
from imblearn.over_sampling import RandomOverSampler
import itertools
from predict import predict_postprocess
import numpy as np
from utils.posprocess import draw_convex_hull
import os

def evaluate(smmodel,backbone,model_path,shape=(320,480)):

    # if shape is None:
    #     shape = (1400,2100)


    train_df, mask_count_df = get_data_preprocessed()

    skf = ShuffleSplit(n_splits=5, test_size=0.15, random_state=133)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index)):


        if n_fold >= 4:
            print('Evaluating fold number ',str(n_fold))


            val_generator = DataGenerator(
                val_indices,
                df=mask_count_df,
                target_df=train_df,
                batch_size=len(val_indices),
                reshape=shape,
                augment=False,
                n_channels=3,
                n_classes=4,
                backbone=backbone
            )

            _ ,y_true = val_generator.__getitem__(0)
            val_generator.batch_size = 8
            # opt = RAdam(lr=0.0002)
            opt = Nadam(lr=0.0002)
            # opt = AdamAccumulate(lr=0.0002, accum_iters=8)

            model = get_model(smmodel,backbone,opt,dice_coef_loss_bce,dice_coef,shape)
            model.load_weights(model_path)

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

            print("Dice: ", np_dice_coef(y_true,y_pred))
            batch_idx = list(range(y_true.shape[0]))
            minsizes = [[5000, 5000, 5000, 5000],
                        [8000, 8000, 8000, 8000],
                        [20000, 20000, 22500, 10000],
                        [10000, 10000, 10000, 10000],
                        [15000, 15000, 15000, 15000],
                        [20000, 20000, 20000, 20000],
                        [25000, 25000, 25000, 25000],
                        [15000, 15000, 10000, 10000],
                        [20000, 15000, 10000, 10000],
                        [20000, 20000, 10000, 10000],
                        [20000, 20000, 15000, 10000],
                        [10000, 20000, 15000, 20000],
                        [10000, 10000, 15000, 15000]]
            thresholds = [0.4,0.45,0.5,0.55,0.6,0.65,0.7]
            for minsize in minsizes:
                for threshold in thresholds:
                    batch_pred_masks = np.array(predict_postprocess(batch_idx, True, y_pred, shape,minsize, threshold))
                    print(minsize)
                    print(threshold)
                    print("Dice with post process: ", np_dice_coef(y_true, np.array(batch_pred_masks)))

            # shape_posprocess_list = ['rect', 'min', 'convex', 'approx']
            #
            # for mode in shape_posprocess_list:
            #     for mask in y_pred:
            #         batch_pred_masks = np.array(draw_convex_hull(mask, mode))
            #         print("Dice with post process: ", np_dice_coef(y_true, np.array(batch_pred_masks)))

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='resnet34', type=str)
    parser.add_argument('--shape', help='Shape of resized images', default=(320, 480), type=tuple)
    parser.add_argument('--model_path', help='model weight path', default=None, type=str)
    parser.add_argument("--cpu", default=False, type=bool)



    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    evaluate(args.model,args.backbone,args.model_path,args.shape)