import numpy as np
import os
import argparse
import sys

def save_blend_int(val_file,type='oof'):

    for i, file in enumerate(os.listdir(val_file)):
        matrix = np.load(val_file + file)
        print(file, matrix.max())
        if i == 0:
            oof_predicted_data = matrix
        else:
            oof_predicted_data += matrix

    oof_predicted_data /= len(os.listdir(val_file))
    print(oof_predicted_data.max())
    print(oof_predicted_data.min())

    # oof_predicted_data = (oof_predicted_data*100).astype(np.int8)
    np.save(val_file + '_blend_' + type +'.npy', oof_predicted_data)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')



    parser.add_argument('--val_file', help='val file to search', default=None, type=str)
    parser.add_argument('--type', help='apply shape convex or not', default='oof', type=str)




    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    save_blend_int(args.val_file,args.type)
