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
from utils.losses import dice_coef, dice_coef_loss_bce, lovasz_loss, combo_loss, np_dice_coef
from utils.callbacks import ValPosprocess
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
from imblearn.over_sampling import RandomOverSampler
import itertools
from predict import predict_postprocess
import numpy as np
from utils.posprocess import draw_convex_hull, post_process_minsize
from utils.utils import mask2rle
import os

def evaluate(smmodel,backbone,model_path,shape=(320,480)):

    # if shape is None:
    #     shape = (1400,2100)


    train_df, mask_count_df = get_data_preprocessed()

    skf = StratifiedKFold(n_splits=5, random_state=133)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):


        if n_fold >= 4:
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
            print(y_pred)

            print("Dice: ", np_dice_coef(y_true,y_pred))
            batch_idx = list(range(y_true.shape[0]))
            minsizes = [[1000, 1000, 1000, 1000],
                        [2000, 2000, 2000, 2000],
                        [3000, 3000, 3000, 3000],
                        [4000, 4000, 4000, 4000],
                        [5000, 5000, 5000, 5000],
                        [6000, 6000, 6000, 6000],
                        [20000, 20000, 22500, 10000]]
            thresholds = [0.58,0.59,0.6,0.61,0.62]
            for minsize in minsizes:
                for threshold in thresholds:
                    batch_pred_masks = np.array(predict_postprocess(batch_idx, True, y_pred, shape,minsize, threshold))
                    print(minsize)
                    print(threshold)
                    print("Dice with post process: ", np_dice_coef(y_true, np.array(batch_pred_masks)))

            shape_posprocess_list = ['rect', 'min', 'convex', 'approx']

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