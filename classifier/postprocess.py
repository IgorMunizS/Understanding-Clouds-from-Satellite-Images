from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from classifier.model import get_model
from sklearn.model_selection import StratifiedKFold
from classifier.preprocess import preprocess
from classifier.generator import DataGenenerator
import os
import pandas as pd
import numpy as np

def get_threshold_for_recall(y_true, y_pred, class_i, recall_threshold=0.94, precision_threshold=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
    i = len(thresholds) - 1
    best_recall_threshold = None
    while best_recall_threshold is None:
        next_threshold = thresholds[i]
        next_recall = recall[i]
        if next_recall >= recall_threshold:
            best_recall_threshold = next_threshold
        i -= 1

    # consice, even though unnecessary passing through all the values
    best_precision_threshold = [thres for prec, thres in zip(precision, thresholds) if prec >= precision_threshold][0]

    return best_recall_threshold, best_precision_threshold


def threshold_search(cls_model='b2', shape=(320,320), submission_file=None):
    class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
    model = get_model(cls_model, shape=shape)
    kfold = StratifiedKFold(n_splits=4, random_state=133, shuffle=True)
    train_df, img_2_vector = preprocess()
    oof_true = []
    oof_pred = []
    for n_fold, (train_indices, val_indices) in enumerate(
            kfold.split(train_df['Image'].values, train_df['Class'].map(lambda x: str(sorted(list(x)))))):
        val_imgs = train_df['Image'].values[val_indices]

        data_generator_val = DataGenenerator(val_imgs, shuffle=False,
                                             resized_height=shape[0], resized_width=shape[1],
                                             img_2_ohe_vector=img_2_vector)

        model.load_weights('checkpoints/' + cls_model + '_' + str(n_fold) + '.h5')

        y_pred = model.predict_generator(data_generator_val, workers=12)
        y_true = data_generator_val.get_labels()

        oof_true.extend(y_true)
        oof_pred.extend(y_pred)


    recall_thresholds = dict()
    precision_thresholds = dict()
    for i, class_name in tqdm(enumerate(class_names)):
        recall_thresholds[class_name], precision_thresholds[class_name] = get_threshold_for_recall(oof_true, oof_pred, i)


    return recall_thresholds


def postprocess_submission():
    data_generator_test = DataGenenerator(folder_imgs='../../dados/test_images', shuffle=False, batch_size=1)
    y_pred_test = model.predict_generator(data_generator_test, workers=12)

    image_labels_empty = set()
    for i, (img, predictions) in enumerate(zip(os.listdir('../../dados/test_images'), y_pred_test)):
        for class_i, class_name in enumerate(class_names):
            if predictions[class_i] < recall_thresholds[class_name]:
                image_labels_empty.add(f'{img}_{class_name}')

    submission = pd.read_csv(submission_file)

    predictions_nonempty = set(submission.loc[~submission['EncodedPixels'].isnull(), 'Image_Label'].values)
    print(f'{len(image_labels_empty.intersection(predictions_nonempty))} masks would be removed')
    submission.loc[submission['Image_Label'].isin(image_labels_empty), 'EncodedPixels'] = np.nan
    submission.to_csv('../submissions/submission_segmentation_and_classifier.csv', index=None)

