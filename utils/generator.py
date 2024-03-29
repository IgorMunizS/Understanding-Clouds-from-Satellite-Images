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
                 gamma=None, TTA=False, randomcrop=False,classes=None):
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
        self.randomcrop = randomcrop
        self.classes = classes
        self.image_name = []

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
        elif self.reshape is not None and self.randomcrop is True:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)

            if self.reshape is not None and self.randomcrop is False:
                img = np_resize(img, self.reshape)

            # Adjust gamma
            if self.gamma is not None:
                img = adjust_gamma(img, gamma=self.gamma)

            # Store samples
            X[i,] = img
            self.image_name.append(im_name)

        return X

    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        elif self.reshape is not None and self.randomcrop is True:
            y = np.empty((self.batch_size, *self.dim,self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            rles = image_df['EncodedPixels'].values

            if self.reshape is not None and self.randomcrop is False:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)

            # background = 1 - masks.sum(axis=-1, keepdims=True)
            # # print(background.shape)
            # masks = np.concatenate((masks,background),axis=-1)
            # # print(masks.shape)

            if self.classes is not None:
                masks = masks[...,self.classes][:,:,None]
            y[i,] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = img.astype(np.float32) / 255.

        except:
            img_name = img_path.split('/')[-1]
            img_path = '../../dados/test_images/' + img_name
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __random_transform(self, img, masks):
        # composition = albu.Compose([
        #     albu.HorizontalFlip(),
        #     albu.VerticalFlip(),
        #     albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15)
        # ])

        composition = albu.Compose([
                        # albu.OneOf([albu.RandomSizedCrop(min_max_height=(self.reshape[0]//2, self.reshape[0]),
                        #                                  height=self.reshape[0], width=self.reshape[1], w2h_ratio=1.5,
                        #                                  p=0.5),
                        #       albu.PadIfNeeded(min_height=self.reshape[0], min_width=self.reshape[1], p=0.5)], p=0.3),
                        # albu.OneOf([
                        #     albu.CropNonEmptyMaskIfExists(self.reshape[0], self.reshape[1]),
                        #     albu.RandomCrop(self.reshape[0],self.reshape[1]),
                        #     albu.Resize(self.reshape[0],self.reshape[1])
                        #
                        # ], p=1.0),
                        albu.RandomSizedCrop(min_max_height=(self.reshape[0] // 2, self.reshape[0]),
                                             height=self.reshape[0], width=self.reshape[1],
                                             w2h_ratio=self.reshape[1] / self.reshape[0], p=0.3),
                        albu.HorizontalFlip(),
                        albu.VerticalFlip(),
                        albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15),
                        albu.OneOf([
                            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            albu.GridDistortion(),
                            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5)
                        ], p=0.3),
                        albu.OneOf([
                            albu.RandomBrightnessContrast(),
                            albu.RandomGamma(),
                            albu.Solarize()
                        ], p=0.5)
        ], p=1)

        composed = composition(image=img.astype(np.uint8), mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def __augment_batch(self, img_batch, masks_batch):
        if self.randomcrop:
            img_batch_crop = np.empty((self.batch_size, *self.reshape, self.n_channels))
            masks_batch_crop = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        for i in range(img_batch.shape[0]):
            if self.randomcrop:
                img_batch_crop[i,], masks_batch_crop[i,] = self.__random_transform(
                    img_batch[i,], masks_batch[i,])
            else:
                img_batch[i,], masks_batch[i,] = self.__random_transform(
                    img_batch[i,], masks_batch[i,])

        if self.randomcrop:
            return img_batch_crop, masks_batch_crop
        else:
            return img_batch, masks_batch

    def do_tta(self,img):

        composition = albu.Compose([
            albu.HorizontalFlip(p=1.),
            albu.VerticalFlip(p=1.),
        ])

        composed = composition(image=img)
        tta_img = composed['image']

        return tta_img