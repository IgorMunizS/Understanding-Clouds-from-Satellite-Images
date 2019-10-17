from keras import callbacks
import numpy as np
from utils.losses import np_dice_coef
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

        val_pred = np.zeros((total, self.shape[0],self.shape[1],4))
        val_true = np.zeros((total,self.shape[0],self.shape[1],4))

        for batch in range(batches):
            xVal, yVal = self.validation_data.__getitem__(batch)
            val_pred[batch * self.batch_size: (batch + 1) * self.batch_size] = np.asarray(
                self.model.predict(xVal)).round()
            val_true[batch * self.batch_size: (batch + 1) * self.batch_size] = yVal

        # val_pred = np.squeeze(val_pred)
        print(val_pred.shape)
        print(val_true.shape)


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


        dice_coef_posprocess = round(np_dice_coef(val_predict_posprocess, val_true), 4)

        print("val_dice_coef_posprocess: {}".format(
                 dice_coef_posprocess))
        return

class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr


    def get_callbacks(self, filepath):

        callback_list = [
            callbacks.ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max',
                                         save_weights_only=True),
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


class SWA(callbacks.Callback):

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                                                  (epoch - self.swa_epoch) + 1)

        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')