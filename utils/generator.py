import keras
import numpy as np
import albumentations as albu
import segmentation_models as sm
import cv2
from utils.utils import build_masks,np_resize
from skimage.exposure import adjust_gamma

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='../../dados/train_images',
                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,
                 augment=False, n_classes=4, random_state=2019, shuffle=True, backbone='resnet34',
                 gamma=None, TTA=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = n_classes
        self.gamma = gamma
        self.shuffle = shuffle
        self.random_state = random_state
        self.preprocess_input = sm.get_preprocessing(backbone)
        self.TTA = TTA


        self.on_epoch_end()
        np.random.seed(self.random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)

            if self.augment:
                X, y = self.__augment_batch(X, y)

            return self.preprocess_input(X), y

        elif self.mode == 'predict':
            return self.preprocess_input(X)

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)

            if self.reshape is not None:
                img = np_resize(img, self.reshape)

            # Adjust gamma
            if self.gamma is not None:
                img = adjust_gamma(img, gamma=self.gamma)

            # Store samples
            X[i,] = img

        return X

    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            rles = image_df['EncodedPixels'].values

            if self.reshape is not None:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)

            y[i,] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img

    def __random_transform(self, img, masks):
        #         composition = albu.Compose([
        #             albu.HorizontalFlip(),
        #             albu.VerticalFlip(),
        #             albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15)
        #         ])

        composition = albu.Compose([
                        albu.OneOf([albu.RandomSizedCrop(min_max_height=(self.reshape[0]//2, self.reshape[0]),
                                                         height=self.reshape[0], width=self.reshape[1], w2h_ratio=1.5,
                                                         p=0.5),
                              albu.PadIfNeeded(min_height=self.reshape[0], min_width=self.reshape[1], p=0.5)], p=1),
                        albu.VerticalFlip(p=0.5),
                        albu.HorizontalFlip(),
                        albu.OneOf([
                            albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            albu.GridDistortion(p=0.5),
                            albu.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
                            ], p=0.8),
                        albu.RandomBrightnessContrast(p=0.8),
                        albu.RandomGamma(p=0.8)])


        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,], masks_batch[i,] = self.__random_transform(
                img_batch[i,], masks_batch[i,])

        return img_batch, masks_batch

    def do_tta(self,img):

        composition = albu.Compose([
            albu.HorizontalFlip(p=1.),
            albu.VerticalFlip(p=1.),
        ])

        composed = composition(image=img)
        tta_img = composed['image']

        return tta_img