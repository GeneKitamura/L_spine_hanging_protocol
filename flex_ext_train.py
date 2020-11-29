import argparse
import pandas as pd

import sys
sys.path.append('..')

import XR_ML_tools.model_tools as model_tools
from XR_ML_tools.fetch_data import prepare_dataset
from XR_ML_tools.fetch_data import TFR_dataset

train_numpy_keras = model_tools.train_numpy_keras
rotate_augment = model_tools.rotate_augment
val_rot_map = model_tools.val_rot_map
position_augment = model_tools.position_augment

# 0 lateral, 1 flex, 2 ext
def npz_dataset():
    dyn_excel = './pos_hard_processed/dyn_3945.xlsx'

    label_col = 'dyn_label'
    train_bool_col = 'new_train'
    train_npz = './dynamic/660_pad_dynto3945.npz'
    val_npz = './dynamic/600_pad_dynto3945.npz'

    datasets = prepare_dataset(dyn_excel, label_col, train_bool_col,
                               train_npz=train_npz, val_npz=val_npz, img_as_uint16=True,
                               use_df_index=False)
    print('using numpy')
    return datasets

def tfr_dataset():
    df = pd.read_excel('./pos_hard_processed/0to7000_dyn.xlsx', index_col='index')
    n_train = df[df['tr_val_te'] == 0].shape[0]
    n_val = df[df['tr_val_te'] == 1].shape[0]
    n_test = df[df['tr_val_te'] == 2].shape[0]
    n_labels = df['dyn_label'].nunique()

    # 6895 from 0 to 2 in NPZ and 6896 from 2 to 1, 6897 from 1 to 0, but hard to change with TFR files
    train_dir = './TFR/train/*'
    val_dir = './TFR/val/*'
    test_dir =  './TFR/test/*'
    datasets = TFR_dataset(train_dir, val_dir, n_train, n_val, n_labels, test_dir=test_dir, n_test=n_test)
    print('using TFR')
    return datasets

def main(epochs, batch_size, save_path, excel_path, use_tfr=True):

    net_input = 600
    if use_tfr:
        datasets = tfr_dataset()
    else:
        datasets = npz_dataset()

    train_numpy_keras(
        datasets,
        epochs=epochs, batch_size=batch_size, save_path=save_path, excel_path=excel_path,
        augment=position_augment(net_input), preprocess_uint16=True, net_input=net_input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        type=str,
        default='./dynamic/600_7000model'
    )
    parser.add_argument(
        '--excel_path',
        type=str,
        default='./dynamic/600_7000.xlsx'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=3
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=150
    )

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS.epochs, FLAGS.batch_size, FLAGS.save_path, FLAGS.excel_path)