import argparse
import sys
from sklearn.model_selection import ShuffleSplit
from utils.preprocess import get_data_preprocessed
from utils.generator import DataGenerator
from keras_radam import RAdam
from keras.optimizers import Adam, Nadam
from utils.lr import CyclicLR
from models import get_model
from utils.losses import dice_coef, dice_coef_loss_bce
from keras.callbacks import ModelCheckpoint, EarlyStopping
import gc


def train(smmodel,backbone,batch_size,shape=(320,480)):


    train_df, mask_count_df = get_data_preprocessed()

    skf = ShuffleSplit(n_splits=5, test_size=0.15, random_state=133)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(mask_count_df.index)):

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
            backbone=backbone
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

        # opt = RAdam(lr=1e-5)
        opt = Nadam(lr=0.0002)

        model = get_model(smmodel,backbone,opt,dice_coef_loss_bce,dice_coef)


        clr = CyclicLR(base_lr=0.0002, max_lr=0.001,
                       step_size=300, reduce_on_plateau=3, monitor='val_loss', reduce_factor=10)

        filepath = '../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=True)
        es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min')
        # rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min', min_delta=0.0001)

        history = model.fit_generator(
            train_generator,
            validation_data=val_generator,
            callbacks=[checkpoint, es, clr],
            epochs=30,
            use_multiprocessing=True,
            workers=42
        )

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




    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)


    train(args.model,args.backbone,args.batch_size,args.shape)