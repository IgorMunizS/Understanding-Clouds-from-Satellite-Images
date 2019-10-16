import multiprocessing
import pandas as pd
import numpy as np


def get_data_preprocessed(pseudo_label=None):
    # remove_image = [
    #     '0031ae9.jpg',
    #     '00b81e1.jpg',
    #     '00cedfa.jpg',
    #     '00dec6a.jpg',
    #     '00dec6a.jpg',
    #     '0107838.jpg',
    #     '01242d7.jpg',
    #     '0153a8b.jpg',
    #     '015aa06.jpg',
    #     '016c135.jpg',
    #     '017ded1.jpg',
    #     '0187cd7.jpg',
    #     '023accd.jpg',
    #     '0269e9a.jpg',
    #     '02c3e33.jpg',
    #     '0326622.jpg',
    #     '03c3906.jpg',
    #     '03ed174.jpg',
    #     '041cc12.jpg',
    #     '042f854.jpg',
    #     '043e76c.jpg',
    #     '0d6d044.jpg',
    #     '100bcc1.jpg',
    #     '150c061.jpg',
    #     '2a77bb5.jpg',
    #     '32362f6.jpg',
    #     '3c46b98.jpg',
    #     '7efc7ea.jpg',
    #     '8825f6a.jpg',
    #     '8acb403.jpg',
    #     '8e8da9f.jpg',
    #     '8f50025.jpg',
    #     '12db338.jpg',
    #     '28ab992.jpg',
    #     '2d6d2f2.jpg',
    #     '332cb5c.jpg',
    #     '389809e.jpg',
    #     '39ee70a.jpg',
    #     '3b14060.jpg',
    #     '3fc6617.jpg',
    #     '449b792.jpg',
    #     '5757384.jpg',
    #     '6521d76.jpg',
    #     '131cad7.jpg',
    #     '1588d4c.jpg',
    #     '24884e7.jpg',
    #     '35c9c03.jpg',
    #     '640441a.jpg',
    #     '7fb5cfc.jpg',
    #     '8c04aab.jpg',
    #     '8db703a.jpg',
    #     '8df3d43.jpg',
    #     'a468e06.jpg',
    #     'afd3da4.jpg',
    #     '046586a.jpg',
    #     '1d03a48.jpg',
    #     '1ecd287.jpg',
    #     '046586a.jpg'
    #     '1588d4c.jpg',
    #     '1e40a05.jpg',
    #     '41f92e5.jpg',
    #     '449b792.jpg',
    #     '563fc48.jpg',
    #     '8bd81ce.jpg',
    #     'c0306e5.jpg',
    #     'c26c635.jpg',
    #     'e04fea3.jpg',
    #     'e5f2f24.jpg',
    #     'eda52f2.jpg',
    #     'fa645da.jpg']


    train_df = pd.read_csv('../../dados/train.csv')

    if pseudo_label is not None:
        pseudo_df = pd.read_csv(pseudo_label)
        pseudo_df = pseudo_df[['Image_Label', 'EncodedPixels']]
        train_df = train_df.append(pseudo_df)

    train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
    train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

    # train_df = train_df[~train_df['ImageId'].isin(remove_image)]


    mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
    mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

    return train_df, mask_count_df


def get_test_data():
    sub_df = pd.read_csv('../../dados/sample_submission.csv')
    sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
    test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

    return sub_df,test_imgs