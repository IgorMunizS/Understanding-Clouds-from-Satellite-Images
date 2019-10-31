import segmentation_models as sm
import UNetPlusPlus.segmentation_models as smx
from utils.jpu import JPU_DeepLab
from deeplabv3.model import Deeplabv3
from tensorflow.keras.optimizers import Nadam
from config import n_classes


def get_model(model,BACKBONE,opt,loss,metric,shape):
    h,w = shape

    if model == 'fpn':
        model = sm.FPN(
            BACKBONE,
            classes=n_classes,
            input_shape=(h, w, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=metric)

    elif model == 'unet':
        model = sm.Unet(
            BACKBONE,
            classes=n_classes,
            input_shape=(h, w, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=metric)

    elif model == 'psp':
        model = sm.PSPNet(
            BACKBONE,
            classes=n_classes,
            input_shape=(h, w, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=metric)

    elif model == 'xnet':
        model = smx.Xnet(
            BACKBONE,
            decoder_block_type='transpose',
            classes=n_classes,
            input_shape=(h, w, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=metric)
    elif model == 'jpu':
        model = JPU_DeepLab(h,w,n_classes)
        model.compile(optimizer=opt, loss=loss, metrics=metric)

    elif model == 'deeplab':
        model = Deeplabv3(weights=None, input_shape=(h,w,3), classes=4, backbone='xception',
              alpha=1., activation='sigmoid')
        model.compile(optimizer=opt, loss=loss, metrics=metric)

    else:
        raise ValueError('Unknown network ' + model)

    return model
