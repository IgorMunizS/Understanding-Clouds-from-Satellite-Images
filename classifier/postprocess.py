class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


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


def postprocess():

    y_pred = model.predict_generator(data_generator_val, workers=num_cores)
    y_true = data_generator_val.get_labels()
    recall_thresholds = dict()
    precision_thresholds = dict()
    for i, class_name in tqdm(enumerate(class_names)):
        recall_thresholds[class_name], precision_thresholds[class_name] = get_threshold_for_recall(y_true, y_pred, i)