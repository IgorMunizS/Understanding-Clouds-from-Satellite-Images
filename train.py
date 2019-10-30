import argparse
import sys
from sklearn.model_selection import StratifiedKFold
from utils.preprocess import get_data_preprocessed
from utils.generator import DataGenerator
from keras_radam import RAdam
from keras.optimizers import Adam, Nadam, SGD
from utils.lr import CyclicLR, Lookahead, AdamAccumulate
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce, sm_loss, lovasz_loss, jaccard,dice_coef_loss
from utils.callbacks import ValPosprocess, SnapshotCallbackBuilder, SWA
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
from imblearn.over_sampling import RandomOverSampler
import itertools
from config import n_fold_splits,random_seed,epochs, ft_epochs
from keras_gradient_accumulation import GradientAccumulation
from segmentation_models.losses import dice_loss

def train(smmodel,backbone,batch_size,shape=(320,480),nfold=0,pseudo_label=None):

    # if shape is None:
    #     shape = (1400,2100)


    train_df, mask_count_df = get_data_preprocessed(pseudo_label)
    ros = RandomOverSampler(random_state=random_seed)

    skf = StratifiedKFold(n_splits=n_fold_splits, random_state=random_seed, shuffle=True)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index, mask_count_df.hasMask)):
        # train_indices, _ = ros.fit_resample(train_indices.reshape(-1, 1),
        #                                            mask_count_df[mask_count_df.index.isin(train_indices)]['hasMask'])
        #
        # train_indices = list(itertools.chain.from_iterable(train_indices))

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
                n_classes=4,
                backbone=backbone,
            )

            val_generator = DataGenerator(
                val_indices,
                df=mask_count_df,
                target_df=train_df,
                batch_size=batch_size,
                reshape=shape,
                augment=False,
                n_channels=3,
                n_classes=4,
                backbone=backbone
            )

            # opt = RAdam(lr=0.0003)
            # opt = Nadam(lr=0.0003)
            opt = Adam(lr=0.0003)
            # opt = AdamAccumulate(lr=0.0003, accum_iters=8)
            # optimizer = GradientAccumulation(opt, accumulation_steps=4)

            dice_focal_loss = sm_loss()
            dice_metric = jaccard()
            model = get_model(smmodel,backbone,opt,dice_coef_loss_bce,[dice_coef,dice_metric],shape)

            filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '.h5'
            ckp = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max',
                                         save_weights_only=True)
            es = EarlyStopping(monitor='val_dice_coef', min_delta=0.0001, patience=5, verbose=1, mode='max')
            rlr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=3, verbose=1, mode='max', min_delta=0.0001)

            history = model.fit_generator(
                train_generator,
                validation_data=val_generator,
                callbacks=[ckp, rlr, es],
                epochs=epochs,
                use_multiprocessing=True,
                workers=42
            )
            # vl_postprocess = ValPosprocess(val_generator,batch_size,shape)
            # lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
            # lookahead.inject(model)
            # snapshot = SnapshotCallbackBuilder(nb_epochs=10, nb_snapshots=1, init_lr=1e-5)
            # callbacks_list = snapshot.get_callbacks(filepath)
            # callbacks_list.append(swa)
            #
            # history = model.fit_generator(
            #     train_generator,
            #     validation_data=val_generator,
            #     callbacks=callbacks_list,
            #     epochs=10,
            #     use_multiprocessing=True,
            #     workers=42
            # )


            # opt = RAdam(lr=0.00001)
            # checkpoint = ModelCheckpoint(filepath, monitor='val_f1-score', verbose=1, save_best_only=True, mode='max',
            #                              save_weights_only=True)
            # # es = EarlyStopping(monitor='val_dice_coef', min_delta=0.0001, patience=5, verbose=1, mode='max')
            # #
            # model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef,dice_metric])
            #
            # clr = CyclicLR(base_lr=0.000001, max_lr=0.00001,
            #                step_size=150, reduce_on_plateau=3, monitor='val_dice_coef', reduce_factor=10, mode='exp_range')
            # swa = SWA('../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '_swa.h5', ft_epochs - 3)
            #
            # history = model.fit_generator(
            #     train_generator,
            #     validation_data=val_generator,
            #     callbacks=[checkpoint, swa, clr],
            #     epochs=ft_epochs,
            #     use_multiprocessing=True,
            #     workers=42
            # )
            #
            # del train_generator,val_generator,model
            # gc.collect()

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