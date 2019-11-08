from classifier.model import get_model
from keras_radam import RAdam
from classifier.generator import DataGenenerator
from sklearn.model_selection import StratifiedKFold
from classifier.preprocess import preprocess
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
from classifier.callbacks import PrAucCallback
import sys
import argparse
import os

def train(model='b2', shape=(320,320)):

    model = get_model(model,shape=shape)
    for base_layer in model.layers[:-3]:
        base_layer.trainable = False

    kfold = StratifiedKFold(n_splits=4, random_state=133, shuffle=True)
    train_df, img_2_vector = preprocess()

    albumentations_train = Compose([
        VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()
    ], p=1)

    for n_fold, (train_imgs, val_imgs) in enumerate(kfold.split(train_df['Image'].values, train_df['Class'].map(lambda x: str(sorted(list(x)))))):

        data_generator_train = DataGenenerator(train_imgs, augmentation=albumentations_train, img_2_ohe_vector=img_2_vector)
        data_generator_train_eval = DataGenenerator(train_imgs, shuffle=False, img_2_ohe_vector=img_2_vector)
        data_generator_val = DataGenenerator(val_imgs, shuffle=False, img_2_ohe_vector=img_2_vector)

        model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        train_metric_callback = PrAucCallback(data_generator_train_eval)
        checkpoint_name = model + '_' + str(n_fold)
        val_callback = PrAucCallback(data_generator_val, stage='val', checkpoint_name=checkpoint_name)

        history_0 = model.fit_generator(generator=data_generator_train,
                                        validation_data=data_generator_val,
                                        epochs=20,
                                        callbacks=[train_metric_callback, val_callback],
                                        workers=42,
                                        verbose=1
                                        )

        for base_layer in model.layers[:-3]:
            base_layer.trainable = True

        model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history_1 = model.fit_generator(generator=data_generator_train,
                                        validation_data=data_generator_val,
                                        epochs=20,
                                        callbacks=[train_metric_callback, val_callback],
                                        workers=42,
                                        verbose=1,
                                        initial_epoch=1
                                        )

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Classification model', default='b2')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--shape', help='Shape of resized images', default=(320,320), type=tuple)
    parser.add_argument("--cpu", default=False, type=bool)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    train(args.model,args.shape)