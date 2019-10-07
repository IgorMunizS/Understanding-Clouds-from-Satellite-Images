import multiprocessing
import pandas as pd
import numpy as np


def get_data_preprocessed():

    train_df = pd.read_csv('../dados/train.csv')
    train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
    train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

    mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
    mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

    return train_df, mask_count_df


def get_test_data():
    sub_df = pd.read_csv('../dados/sample_submission.csv')
    sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
    test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

    return test_imgs