#!/usr/bin/env python

import argparse
import sys
sys.path.append('..')

import XR_ML_tools.model_tools as model_tools

from XR_ML_tools.fetch_data import prepare_dataset

train_numpy_keras = model_tools.train_numpy_keras

# pos_label = {0: ap, 1: ll, 2: lo, 3: ls, 4: rl, 5: ro, 6: rs, 7:bad}
def pos0to1495_uint8(get_pos, epochs, batch_size, save_path, excel_path):
    print('get_pos is {}'.format(get_pos))
    df_excel ='./pos_hard_processed/labels_df.xlsx'

    if get_pos:
        label_col = 'new_pos_label'
        train_bool_col = 'pos_train'
    else:
        label_col = 'hard_label'
        train_bool_col = 'hard_train'

    train_numpy_keras(prepare_dataset(df_excel, label_col, train_bool_col), epochs=epochs, batch_size=batch_size, save_path=save_path, excel_path=excel_path)

def pos0to7000_uint16(get_pos, epochs, batch_size, save_path, excel_path):
    print('Traing for {}'.format(get_pos))
    train_npz = './pos_hard_npz/238_0to7000.npz'
    val_npz = './pos_hard_npz/224_0to7000.npz'
    img_as_uint16=True
    test_set_col = 'tr_val_te'

    if get_pos == 'pos':
        df_excel ='./pos_hard_processed/pos_correct0to7000.xlsx'
        label_col = 'true_label'
        train_bool_col = 'new_train'
    else:
        df_excel = './pos_hard_processed/hard_correctedto7000.xlsx'
        label_col = 'hard_label'
        train_bool_col = 'new_train'

    dataset = prepare_dataset(df_excel, label_col, train_bool_col,
                              train_npz=train_npz, val_npz=val_npz,
                              img_as_uint16=img_as_uint16, test_set_col=test_set_col)

    train_numpy_keras(dataset, preprocess_uint16=img_as_uint16, epochs=epochs,
                      batch_size=batch_size, save_path=save_path, excel_path=excel_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pos_or_hard',
        type=str,
        choices=['pos', 'hard'],
        default='pos' # bool doesn't work for argsparse
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./pos_hard_outs/pos7000'
    )
    parser.add_argument(
        '--excel_path',
        type=str,
        default='./pos_hard_outs/pos7000.xlsx'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100
    )

    FLAGS, _ = parser.parse_known_args()
    pos0to7000_uint16(FLAGS.pos_or_hard, FLAGS.epochs, FLAGS.batch_size, FLAGS.save_path, FLAGS.excel_path)