from keras import callbacks
import numpy as np
from utils.losses import dice_coef
import cv2
from utils.posprocess import post_process

def post_process_callback(val_predict,shape):
    minsizes = [20000, 20000, 22500, 10000]

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    for j, b in enumerate(len(val_predict)):

        pred_masks = val_predict[j,]
        arrt = np.array([])
        for t in range(4):
            a, num_predict = post_process(sigmoid(pred_masks[:, :, t]), 0.6, minsizes[t], shape)

            if (arrt.shape == (0,)):
                arrt = a.reshape(shape[0], shape[1], 1)
            else:
                arrt = np.append(arrt, a.reshape(shape[0], shape[1], 1), axis=2)

        pred_masks = arrt

        return pred_masks


class ValPosprocess(callbacks.Callback):

    def __init__(self, val_data, batch_size=20,shape=(320,480)):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.shape = shape


    def on_epoch_end(self, epoch, logs={}):

        batches = len(self.validation_data)
        total = batches * self.batch_size

        val_pred = np.zeros((total, 1))
        val_true = np.zeros((total))

        for batch in range(batches):
            xVal, yVal = self.validation_data.__getitem__(batch)
            val_pred[batch * self.batch_size: (batch + 1) * self.batch_size] = np.asarray(
                self.model.predict(xVal)).round()
            val_true[batch * self.batch_size: (batch + 1) * self.batch_size] = yVal

        val_pred = np.squeeze(val_pred)



        # # 5.4.1 For each validation batch
        # for batch_index in range(0, len(self.validation_data)):
        #     # 5.4.1.1 Get the batch target values
        #     temp_targ = self.validation_data[batch_index][1]
        #     # 5.4.1.2 Get the batch prediction values
        #     temp_predict = (np.asarray(self.model.predict(
        #                         self.validation_data[batch_index][0]))).round()
        #     # 5.4.1.3 Append them to the corresponding output objects
        #     if(batch_index == 0):
        #         val_targ = temp_targ
        #         val_predict = temp_predict
        #     else:
        #         val_targ = np.vstack((val_targ, temp_targ))
        #         val_predict = np.vstack((val_predict, temp_predict))

        val_predict_posprocess = post_process_callback(val_pred,self.shape)


        dice_coef_posprocess = round(dice_coef(val_predict_posprocess, val_predict), 4)

        print("val_dice_coef_posprocess: {}".format(
                 dice_coef_posprocess))
        return