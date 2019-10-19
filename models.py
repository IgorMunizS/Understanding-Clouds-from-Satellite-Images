import segmentation_models as sm
import UNetPlusPlus.segmentation_models as smx
from utils.jpu import JPU_DeepLab
from tensorflow.keras.optimizers import Nadam

def get_model(model,BACKBONE,opt,loss,metric,shape):
    h,w = shape

    if model == 'fpn':
        model = sm.FPN(
            BACKBONE,
            classes=4,
            input_shape=(h, w, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=[metric])

    elif model == 'unet':
        model = sm.Unet(
            BACKBONE,
            classes=4,
            input_shape=(h, w, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=[metric])

    elif model == 'xnet':
        model = smx.Xnet(
            BACKBONE,
            decoder_block_type='transpose',
            classes=4,
            input_shape=(h, w, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=[metric])
    elif model == 'jpu':
        model = JPU_DeepLab(h,w,4)
        model.compile(optimizer=opt, loss=loss, metrics=[metric])
    else:
        raise ValueError('Unknown network ' + model)

    return model
