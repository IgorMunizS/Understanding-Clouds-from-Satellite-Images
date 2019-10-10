from keras import callbacks
import numpy as np
from utils.losses import dice_coef
import cv2
from utils.posprocess import post_process

def post_process_callback(val_predict,shape):
    minsizes = [20000, 20000, 22500, 10000]
    val_predict_posprocess = np.zeros(shape=val_predict.shape)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    for j, pred_masks in enumerate(val_predict):

        arrt = np.array([])
        for t in range(4):
            a, num_predict = post_process(sigmoid(pred_masks[:, :, t]), 0.6, minsizes[t], shape)

            if (arrt.shape == (0,)):
                arrt = a.reshape(shape[0], shape[1], 1)
            else:
                arrt = np.append(arrt, a.reshape(shape[0], shape[1], 1), axis=2)

        val_predict_posprocess[j,] = arrt

        return val_predict_posprocess


class ValPosprocess(callbacks.Callback):

    def __init__(self, val_data, batch_size=20,shape=(320,480)):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.shape = shape


    def on_epoch_end(self, epoch, logs={}):

        batches = len(self.validation_data)
        total = batches * self.batch_size
        dice_coef_batchs = []

        # val_pred = np.zeros((total, self.shape[0],self.shape[1],4))
        # val_true = np.zeros((total,self.shape[0],self.shape[1],4))

        for batch in range(batches):
            xVal, yVal = self.validation_data.__getitem__(batch)
            val_pred = np.asarray(
                self.model.predict(xVal)).round()


            val_predict_posprocess = post_process_callback(val_pred, self.shape)

            dice_coef_batchs.append(round(dice_coef(yVal.astype('float32'),val_predict_posprocess.astype('float32')), 4))

        dice_coef_posprocess = np.mean(dice_coef_batchs,axis=0)

        # val_pred = np.squeeze(val_pred)



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



        print("val_dice_coef_posprocess: {}".format(
                 dice_coef_posprocess))
        return