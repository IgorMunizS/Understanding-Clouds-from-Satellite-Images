import segmentation_models as sm

def get_model(model,BACKBONE,opt,loss,metric):

    if model == 'fpn':
        model = sm.FPN(
            BACKBONE,
            classes=4,
            input_shape=(320, 480, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=[metric])

    elif model == 'unet':
        model = sm.Unet(
            BACKBONE,
            classes=4,
            input_shape=(320, 480, 3),
            activation='sigmoid'
        )
        model.compile(optimizer=opt, loss=loss, metrics=[metric])
    else:
        raise ValueError('Unknown network ' + model)

    return model
