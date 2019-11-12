import argparse
import sys
from sklearn.model_selection import StratifiedKFold
from utils.preprocess import get_data_preprocessed
from utils.generator import DataGenerator
from keras_radam import RAdam
from keras.optimizers import Adam, Nadam, SGD, RMSprop
from utils.lr import CyclicLR, Lookahead, AdamAccumulate
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce, dice_coef_fish,dice_coef_flower,dice_coef_gravel,dice_coef_sugar
from utils.losses import jaccard, sm_loss, combo_loss_init, combo_loss_ft, tversky_loss
from utils.callbacks import ValPosprocess, SnapshotCallbackBuilder, SWA
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
from imblearn.over_sampling import RandomOverSampler
import itertools
from config import n_fold_splits,random_seed,epochs, n_classes,classes,ft_epochs
from keras_gradient_accumulation import GradientAccumulation
from segmentation_models.losses import dice_loss

def train(smmodel,backbone,batch_size,shape=(320,480),nfold=0,pseudo_label=None):

    # if shape is None:
    #     shape = (1400,2100)


    train_df, mask_count_df = get_data_preprocessed(pseudo_label)
    ros = RandomOverSampler(random_state=random_seed)

    skf = StratifiedKFold(n_splits=n_fold_splits, random_state=random_seed,  shuffle=True)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):
        # train_indices, _ = ros.fit_resample(train_indices.reshape(-1, 1),
        #                                            mask_count_df[mask_count_df.index.isin(train_indices)]['hasMask'])
        #
        # train_indices = list(itertools.chain.from_iterable(train_indices))
        # val_images = mask_count_df[mask_count_df.index.isin(val_indices)]['ImageId'].tolist()
        # train_images = mask_count_df[mask_count_df.index.isin(train_indices)]['ImageId'].tolist()
        # #
        # for classe in ['Fish','Flower','Gravel','Sugar']:
        #     train_df, mask_count_df = get_data_preprocessed(pseudo_label, classe=classe)
        #     val_indices = mask_count_df[mask_count_df['ImageId'].isin(val_images)].index
        #     train_indices = mask_count_df[mask_count_df['ImageId'].isin(train_images)].index
        #     print('Training class ', classe)
        if n_fold >= nfold:
            print('Training fold number ',str(n_fold))


            train_generator = DataGenerator(
                train_indices,
                df=mask_count_df,
                target_df=train_df,
                batch_size=batch_size,
                reshape=shape,
                augment=True,
                n_channels=3,
                n_classes=n_classes,
                backbone=backbone,
                randomcrop=False,
            )

            val_generator = DataGenerator(
                val_indices,
                df=mask_count_df,
                target_df=train_df,
                batch_size=batch_size,
                reshape=shape,
                augment=False,
                n_channels=3,
                n_classes=n_classes,
                backbone=backbone,
                randomcrop=False,
            )

            # opt = RAdam(lr=0.0003)
            opt = Nadam(lr=0.0003)
            # opt = RMSprop(lr=0.0003)
            # accum_iters = 64 // batch_size
            # opt = AdamAccumulate(lr=0.0003, accum_iters=accum_iters)
            # optimizer = GradientAccumulation(opt, accumulation_steps=4)

            dice_focal_loss = sm_loss()
            dice_metric = jaccard()

            metrics = [dice_coef,dice_coef_fish,dice_coef_flower,dice_coef_gravel,dice_coef_sugar]
            model = get_model(smmodel,backbone,opt,tversky_loss,[dice_coef])
            filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold)
            # filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '_' + classe

            ckp = ModelCheckpoint(filepath + '.h5', monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max',
                                         save_weights_only=True)

            es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min')
            rlr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=3, verbose=1, mode='max', min_delta=0.0001)

            history = model.fit_generator(
                train_generator,
                validation_data=val_generator,
                callbacks=[ckp, rlr, es],
                epochs=epochs,
                use_multiprocessing=True,
                workers=42
            )

            # opt = RAdam(min_lr=1e-6, lr=1e-5)
            # # model.compile(optimizer=opt, loss=combo_loss_ft, metrics=[dice_coef])
            # model2 = get_model(smmodel,backbone,opt,lovasz_loss,[dice_coef],freeze_encoder=True,batchnormalization=False)
            # model2.set_weights(model.get_weights())
            # es = EarlyStopping(monitor='val_dice_coef', min_delta=0.0001, patience=3, verbose=1, mode='max')
            # del model
            # gc.collect()
            # history = model2.fit_generator(
            #     train_generator,
            #     validation_data=val_generator,
            #     callbacks=[ckp, es],
            #     epochs=ft_epochs,
            #     use_multiprocessing=True,
            #     workers=42
            # )

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='resnet34', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--shape', help='Shape of resized images', default=(320,480), type=tuple)
    parser.add_argument('--n_fold', help='Number of fold to start training', default=0, type=int)
    parser.add_argument('--pseudo_label', help='Add extra data from test', default=None, type=str)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)


    train(args.model,args.backbone,args.batch_size,args.shape,args.n_fold,args.pseudo_label)