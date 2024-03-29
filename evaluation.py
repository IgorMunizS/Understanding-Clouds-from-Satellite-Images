import argparse
import sys
from sklearn.model_selection import StratifiedKFold
from utils.preprocess import get_data_preprocessed
from utils.generator import DataGenerator
from keras.optimizers import Adam, Nadam, SGD
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce, lovasz_loss, combo_loss, np_dice_coef,dice
import gc
from utils.posprocess import post_process
import ray
import psutil
import numpy as np
from utils.posprocess import draw_convex_hull, post_process_minsize
from utils.utils import mask2rle, np_resize
import os
import pandas as pd
import time
from tqdm import tqdm
from tta_wrapper import tta_segmentation
from config import n_fold_splits, random_seed, n_classes
import os

def resize_oof(folder):
    files = os.listdir(folder)
    for file in files:
        print("Resizing file: ", file)
        oof_data = np.load(folder + file)
        new_oof_data = np.zeros((oof_data.shape[0], 350,525,4), dtype=np.float16)
        for i in range(oof_data.shape[0]):
            new_oof_data[i,:,:,:] = np_resize(oof_data[i,:,:,:].astype(np.float32), (350,525)).astype(np.float16)
        np.save(folder + file, new_oof_data)

def convert_to_int(folder):
    files = os.listdir(folder)
    for file in files:
        print("Resizing file: ", file)
        oof_data = np.load(folder + file)
        new_oof_data = (oof_data*100).astype(np.int8)
        np.save(folder + file, new_oof_data)

# @ray.remote
def parallel_post_process(y_true,y_pred,class_id,t,ms,bt,shape,fixshape):
    # sigmoid = lambda x: 1 / (1 + np.exp(-x))

    masks = []
    for i in range(y_pred.shape[0]):
        probability = y_pred[i, :, :, class_id].astype(np.float32)
        predict, num_predict = post_process(probability, t, ms,bt, shape,fixshape)
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

def multimodel_eval(smmodel,backbone,nfold,maxfold,shape=(320,480),swa=False, tta=False,fixshape=True):
    h,w =shape


    train_df, mask_count_df = get_data_preprocessed()
    opt = Nadam(lr=0.0002)

    skf = StratifiedKFold(n_splits=n_fold_splits, random_state=random_seed, shuffle=True)
    oof_data = np.zeros((len(mask_count_df.index), 350, 525, 4), dtype=np.float16)
    oof_predicted_data = np.zeros((len(mask_count_df.index), 350, 525, 4), dtype=np.float16)
    oof_imgname = []

    oof_dice = []
    classes=['Fish','Flower','Gravel','Sugar']
    cnt_position = 0

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):
        final_pred = np.zeros((len(val_indices),h,w,4), dtype=np.float32)
        y_true = []
        if n_fold >= nfold and n_fold <= maxfold:
            if n_fold >= 2:
                nclass = 4
                print(nclass)
            else:
                nclass = n_classes

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
            img_true = val_generator.image_name
            oof_imgname.extend(img_true)
            val_generator.batch_size = 1

            for i,cls in enumerate(classes):

                model = get_model(smmodel, backbone, opt, dice_coef_loss_bce, [dice_coef], nclass=nclass)

                filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '_' + cls

                if swa:
                    filepath += '_swa.h5'
                else:
                    filepath += '.h5'


                model.load_weights(filepath)

                # results = model.evaluate_generator(
                #     val_generator,
                #     workers=40,
                #     verbose=1
                # )
                # print(results)

                if tta:
                    model = tta_segmentation(model, h_flip=True, v_flip=True,
                                         input_shape=(h, w, 3), merge='mean')


                y_pred = model.predict_generator(
                    val_generator,
                    workers=40,
                    verbose=1
                    )

                final_pred[:,:,:,i] = y_pred[:,:,:,0]

                del y_pred
                gc.collect()

            print(y_true.shape)
            print(final_pred.shape)
            print(len(oof_imgname))
            d = np_dice_coef(y_true, final_pred)
            oof_dice.append(d)
            print("Dice: ", d)

            for i in range(y_true.shape[0]):
                oof_data[cnt_position, :, :, :] = np_resize(y_true[i, :, :, :].astype(np.float32), (350, 525)).astype(
                    np.float16)
                oof_predicted_data[cnt_position, :, :, :] = np_resize(final_pred[i, :, :, :].astype(np.float32),
                                                                      (350, 525)).astype(np.float16)
                cnt_position += 1

        del y_true, final_pred
        gc.collect()

    del val_generator, model
    gc.collect()

    oof_imgname = np.asarray(oof_imgname)
    print(oof_data.shape)
    print(oof_predicted_data.shape)
    print(oof_imgname.shape)
    print("CV Final Dice: ", np.mean(oof_dice))

    np.save('../validations/img_name_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold_splits) + '.npy', oof_imgname)
    np.save('../validations/y_true_' + str(n_fold_splits) + '.npy', oof_data)
    np.save('../validations/' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold_splits) + '.npy', oof_predicted_data)


def evaluate(smmodel,backbone,nfold,maxfold,shape=(320,480),swa=False, tta=False,fixshape=True):
    h,w =shape


    train_df, mask_count_df = get_data_preprocessed()
    opt = Nadam(lr=0.0002)

    skf = StratifiedKFold(n_splits=n_fold_splits, random_state=random_seed, shuffle=True)
    oof_data = np.zeros((len(mask_count_df.index), 350,525,4), dtype=np.float16)
    oof_predicted_data = np.zeros((len(mask_count_df.index), 350,525,4), dtype=np.float16)
    oof_imgname = []
    # num_cpus = psutil.cpu_count(logical=False)
    # ray.init(num_cpus=4)
    oof_dice = []
    cnt_position=0
    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):


        for i in range(0, len(val_indices), 480):
            batch_idx = list(
                range(i, min(len(val_indices), i + 480))
            )
            model = get_model(smmodel, backbone, opt, dice_coef_loss_bce, [dice_coef])
            if n_fold >= nfold and n_fold <= maxfold:
                print('Evaluating fold number ',str(n_fold))

                val_generator = DataGenerator(
                    val_indices[batch_idx],
                    df=mask_count_df,
                    shuffle=False,
                    target_df=train_df,
                    batch_size=len(val_indices[batch_idx]),
                    reshape=shape,
                    augment=False,
                    n_channels=3,
                    n_classes=4,
                    backbone=backbone
                )

                _ ,y_true = val_generator.__getitem__(0)
                img_true = val_generator.image_name
                oof_imgname.extend(img_true)
                val_generator.batch_size = 1

                filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold)

                if swa:
                    filepath += '_swa.h5'
                else:
                    filepath += '.h5'


                model.load_weights(filepath)

                # results = model.evaluate_generator(
                #     val_generator,
                #     workers=40,
                #     verbose=1
                # )
                # print(results)

                if tta:
                    model = tta_segmentation(model, h_flip=True, v_flip=True,
                                             input_shape=(h, w, 3), merge='mean')


                y_pred = model.predict_generator(
                    val_generator,
                    workers=40,
                    verbose=1
                )
                print(y_true.shape)
                print(y_pred.shape)
                print(len(oof_imgname))


                f=[]
                for i in range(y_true.shape[0]):
                    for j in range(4):
                        d1 = np_dice_coef(y_true[i,:,:,j], y_pred[i,:,:,j])
                        f.append(d1)

                    oof_data[cnt_position,:,:,:] = np_resize(y_true[i,:,:,:].astype(np.float32), (350,525)).astype(np.float16)
                    oof_predicted_data[cnt_position,:,:,:] = np_resize(y_pred[i,:,:,:].astype(np.float32), (350,525)).astype(np.float16)
                    cnt_position +=1
                print("Dice: ", np.mean(f))
                oof_dice.append(np.mean(f))
                del y_true, y_pred, val_generator, model
                gc.collect()

    # oof_data = np.asarray(oof_data)
    # oof_predicted_data = np.asarray(oof_predicted_data)
    oof_imgname = np.asarray(oof_imgname)
    print(oof_data.shape)
    print(oof_predicted_data.shape)
    print(oof_imgname.shape)
    print("CV Final Dice: ", np.mean(oof_dice))

    np.save('../validations/img_name_' + str(n_fold_splits) + '.npy', oof_imgname)
    np.save('../validations/y_true_' + str(n_fold_splits) + '.npy', oof_data)
    np.save('../validations/' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold_splits) + '.npy', oof_predicted_data)

    # now = time.time()
    # class_params = {}
    # for class_id in range(4):
    #     print(class_id)
    #     attempts = []
    #     for t in tqdm(range(40, 80, 5)):
    #         t /= 100
    #         for ms in tqdm(range(10000, 31000, 5000)):
    #
    #             d = parallel_post_process(oof_data,oof_predicted_data,class_id,t,ms,None,shape,fixshape)
    #
    #             # print(t, ms, np.mean(d))
    #             attempts.append((t, ms, np.mean(d)))
    #
    #     attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
    #
    #     attempts_df = attempts_df.sort_values('dice', ascending=False)
    #     print(attempts_df.head())
    #     print('Time: ', time.time() - now)
    #     best_threshold = attempts_df['threshold'].values[0]
    #     best_size = attempts_df['size'].values[0]
    #     class_params[class_id] = (best_threshold, best_size)


def search(val_file,shape,classid=0,fixshape=False, emsemble=False, yves=False):

    h,w = (350,525)

    oof_data = np.load('../validations/y_true_' + str(n_fold_splits) + '.npy')

    if emsemble:
        for i,file in enumerate(os.listdir(val_file)):
            if i == 0:
                oof_predicted_data = np.load(val_file + file)
            else:
                oof_predicted_data += np.load(val_file + file)

        oof_predicted_data /= len(os.listdir(val_file))
    else:
        oof_predicted_data = np.load(val_file)

    print(oof_data.shape)
    print(oof_predicted_data.shape)

    #search range variables
    min_thre = 55
    max_thre = 81
    min_minsize = 0
    max_minsize = 21000
    min_bottom = 40

    if yves:
        min_thre += 15
        max_thre += 5
        min_minsize += 7500
        max_minsize += 7500
        min_bottom +=25

        for x,img in tqdm(enumerate(oof_predicted_data)):
            for k in range(4):
                im_layer = img[:,:,k]
                max_col = np.max(im_layer, axis=1)
                max_row = np.max(im_layer, axis=0)
                mat = [max_col[i] + max_row[j] for i in range(h) for j in range(w)]
                mat = np.reshape(mat, (h,w))
                im_layer = 0.7*im_layer + 0.3*mat

                oof_predicted_data[x,:,:,k] = im_layer


    now = time.time()
    class_params = {}
    for class_id in range(4):
        print(class_id)
        if class_id >= classid:
            attempts = []
            for t in tqdm(range(min_thre, max_thre, 5)): #threshold post process
                t /= 100
                for ms in tqdm(range(min_minsize, max_minsize, 5000)): #minsize post process
                    for bt in range(min_bottom, int(t*100 - 1), 5): #bottom threshold
                        bt /= 100

                        d = parallel_post_process(oof_data,oof_predicted_data,class_id,t,ms,bt,(350,525),fixshape)

                        # print(t, ms, np.mean(d))
                        attempts.append((t, ms, bt, np.mean(d)))

            attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'bottom', 'dice'])

            attempts_df = attempts_df.sort_values('dice', ascending=False)
            print(attempts_df.head())
            print('Time: ', time.time() - now)
            best_threshold = attempts_df['threshold'].values[0]
            best_size = attempts_df['size'].values[0]
            best_bottom = attempts_df['bottom'].values[0]
            best_dice = attempts_df['dice'].values[0]
            class_params[class_id] = (best_threshold, best_size, best_bottom,best_dice)

    print(class_params)
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
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='resnet34', type=str)
    parser.add_argument('--shape', help='Shape of resized images', nargs='+', default=[320, 480], type=int)
    parser.add_argument('--nfold', help='number of fold to evaluate', default=0, type=int)
    parser.add_argument('--maxfold', help='number of fold to evaluate', default=5, type=int)
    parser.add_argument('--tta', help='apply TTA', default=False, type=bool)
    parser.add_argument('--swa', help='apply SWA', default=False, type=bool)
    parser.add_argument('--search', help='search post processing values', default=False, type=bool)
    parser.add_argument('--classid', help='class id to start search', default=0, type=int)
    parser.add_argument('--val_file', help='val file to search', default=None, type=str)
    parser.add_argument('--fixshape', help='apply shape convex or not', default=False, type=bool)
    parser.add_argument('--yves', help='apply shape yves', default=False, type=bool)
    parser.add_argument('--emsemble', help='Validation emsemble of models. val_file must be a path to folder with all models', default=False, type=bool)
    parser.add_argument('--multimodel', help='Multi class model', default=False, type=bool)
    parser.add_argument('--resize', help='Resize oof datal', default=False, type=bool)
    parser.add_argument('--folder', help='folder to resize data', default=None, type=str)

    parser.add_argument("--cpu", default=False, type=bool)



    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    h,w = args.shape
    newshape = (h,w)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.search:
        search(args.val_file,newshape,args.classid,args.fixshape, args.emsemble, args.yves)
    elif args.multimodel:
        multimodel_eval(args.model,args.backbone,args.nfold,args.maxfold,newshape,args.swa,args.tta,args.fixshape)
    elif args.resize:
        resize_oof(args.folder)
    else:
        evaluate(args.model,args.backbone,args.nfold,args.maxfold,newshape,args.swa,args.tta,args.fixshape)