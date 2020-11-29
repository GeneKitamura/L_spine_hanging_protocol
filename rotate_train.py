#!/usr/bin/env python

import tensorflow as tf
import argparse

import sys
sys.path.append('..')

import XR_ML_tools.model_tools as model_tools
from XR_ML_tools.fetch_data import prepare_dataset

train_numpy_keras = model_tools.train_numpy_keras
rotate_augment = model_tools.rotate_augment
val_rot_map = model_tools.val_rot_map

# pos_label = {0: ap, 1: ll, 2: lo, 3: ls, 4: rl, 5: ro, 6: rs, 7: bad}
def main(epochs, batch_size, save_path, excel_path, rot_degree):
    train_npz = './pos_hard_npz/238_0to7000.npz'
    val_npz = './pos_hard_npz/224_0to7000.npz'

    df_excel = './pos_hard_processed/rot0to7000.xlsx'
    true_labels = './pos_hard_processed/pos_correct0to7000.xlsx'
    test_set_col = 'tr_val_te'
    flip_RtoL_col = 'true_label'

    label_col = 'angle'
    train_bool_col = 'new_train'
    rand_rot_angle = rot_degree
    Loss_MAE=tf.keras.losses.MeanAbsoluteError()
    Loss_MSE=tf.keras.losses.MeanSquaredError()
    Metric_MAE = tf.keras.metrics.MeanAbsoluteError()
    Metric_MSE = tf.keras.metrics.MeanSquaredError()

    train_numpy_keras(prepare_dataset(df_excel, label_col, train_bool_col,
                                      train_npz=train_npz, val_npz=val_npz,
                                      img_as_uint16=True, flip_RtoL_col=flip_RtoL_col, npz_df=true_labels, test_set_col=test_set_col),
                      epochs=epochs, batch_size=batch_size, save_path=save_path, excel_path=excel_path,
                      augment=rotate_augment(rand_rot_angle), val_map=val_rot_map(rand_rot_angle), preprocess_uint16=True,
                      n_class=1, activation=tf.keras.activations.tanh,
                      loss=Loss_MAE, metrics=[Metric_MAE, Metric_MSE],
                      monitor='loss')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        type=str,
        default='./rot/model_7000_100deg'
    )
    parser.add_argument(
        '--excel_path',
        type=str,
        default='./rot/rot_out_7000_100deg.xlsx'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=120
    )
    parser.add_argument(
        '--rot_degree',
        type=int,
        default=100
    )

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS.epochs, FLAGS.batch_size, FLAGS.save_path, FLAGS.excel_path, FLAGS.rot_degree)