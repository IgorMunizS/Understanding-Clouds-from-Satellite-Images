import cv2
import numpy as np

def post_process(probability, threshold, min_size,bottom_threshold,shape=(350, 525), fixshape=False):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    if fixshape:
        mask = draw_convex_hull(mask.astype(np.uint8), mode='convex')

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(shape, np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1

    if predictions.sum() > 0 and bottom_threshold is not None:
        predictions = cv2.threshold(probability, bottom_threshold, 1, cv2.THRESH_BINARY)[1]

    return predictions, num


def draw_convex_hull(mask, mode='convex'):
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == 'rect':  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        elif mode == 'convex':  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255), -1)
        elif mode == 'approx':
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255), -1)
        else:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
    return img / 255.


def post_process_minsize(mask, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(mask.shape)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions  # , num