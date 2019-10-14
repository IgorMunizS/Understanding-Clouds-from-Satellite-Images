import argparse
import sys
from sklearn.model_selection import ShuffleSplit
from utils.preprocess import get_data_preprocessed
from utils.generator import DataGenerator
from keras_radam import RAdam
from keras.optimizers import Adam, Nadam, SGD
from utils.lr import CyclicLR, Lookahead, AdamAccumulate
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce, lovasz_loss, combo_loss
from utils.callbacks import ValPosprocess
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
from segmentation_models.losses import bce_jaccard_loss


def train(smmodel,backbone,batch_size,shape=(320,480),nfold=0):

    # if shape is None:
    #     shape = (1400,2100)


    train_df, mask_count_df = get_data_preprocessed()

    skf = ShuffleSplit(n_splits=5, test_size=0.15, random_state=133)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index)):

        if n_fold >= nfold:
            print('Training fold number ',str(n_fold))


            train_generator = DataGenerator(
                train_indices,
                df=mask_count_df,
                target_df=train_df,
                batch_size=batch_size,
                reshape=None,
                augment=True,
                n_channels=3,
                n_classes=4,
                backbone=backbone
            )

            val_generator = DataGenerator(
                val_indices,
                df=mask_count_df,
                target_df=train_df,
                batch_size=batch_size,
                reshape=None,
                augment=True,
                n_channels=3,
                n_classes=4,
                backbone=backbone
            )

            # opt = RAdam(lr=0.0002)
            opt = Nadam(lr=0.0002)
            # opt = AdamAccumulate(lr=0.0001, accum_iters=5)

            model = get_model(smmodel,backbone,opt,dice_coef_loss_bce,dice_coef,shape)


            filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '.h5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                         save_weights_only=True)
            es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min')
            rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min', min_delta=0.0001)

            # vl_postprocess = ValPosprocess(val_generator,batch_size,shape)
            # lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
            # lookahead.inject(model)

            history = model.fit_generator(
                train_generator,
                validation_data=val_generator,
                callbacks=[checkpoint, es, rlr],
                epochs=30,
                use_multiprocessing=True,
                workers=42
            )

            # opt = RAdam(lr=0.00001)
            # checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max',
            #                              save_weights_only=True)
            # es = EarlyStopping(monitor='val_dice_coef', min_delta=0.0001, patience=5, verbose=1, mode='max')
            #
            # model.compile(optimizer=opt, loss=combo_loss, metrics=[dice_coef])
            #
            # clr = CyclicLR(base_lr=0.000001, max_lr=0.00001,
            #                step_size=150, reduce_on_plateau=3, monitor='val_dice_coef', reduce_factor=10, mode='exp_range')
            #
            # history = model.fit_generator(
            #     train_generator,
            #     validation_data=val_generator,
            #     callbacks=[checkpoint, es, clr],
            #     epochs=30,
            #     use_multiprocessing=True,
            #     workers=42
            # )

            del train_generator,val_generator,model
            gc.collect()

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='resnet34', type=str)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--shape', help='Shape of resized images', default=(320,480), type=tuple)
    parser.add_argument('--n_fold', help='Number of fold to start training', default=0, type=int)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)


    train(args.model,args.backbone,args.batch_size,args.shape,args.n_fold)