import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skimage
import tensorflow as tf
import math

from XR_ML_tools import fetch_data

import sys
sys.path.append('..')

import XR_ML_tools.utils as utils
import XR_ML_tools.model_tools as model_tools

split_train_val_df = model_tools.split_train_val_df
trunc_name = model_tools.trunc_name
rotate_augment = model_tools.rotate_augment
val_rot_map = model_tools.val_rot_map
pass_through = model_tools.pass_through
position_augment = model_tools.position_augment
preprocess_densenet = model_tools.preprocess_densenet
data_get = fetch_data.data_get
prepare_dataset = fetch_data.prepare_dataset


def check_rotation():
    IMG_ROOT = '../L_spine_images/Images/'
    MAP_FILE = './../PHI/DEXA/R3_1750_PT_MAP_FINAL.xlsx'
    R3_REQ = './../PHI/DEXA/R1750.xlsx'
    USER_XLSX = "../PHI/DEXA/L_spine_consolidated.xlsx"

    short_full_merged = pd.read_excel('../PHI/DEXA/short_full_merged.xlsx', index_col='index')

    df_excel ='./pos_hard_processed/rot_labels.xlsx'
    label_col = 'angle'
    train_bool_col = 'train'
    batch_size=20

    df = pd.read_excel(df_excel, index_col='index')

    augment=rotate_augment
    val_map=val_rot_map
    preprocess_map=preprocess_densenet
    preprocess_uint16=False
    inner_preprocess = preprocess_map(uint16=preprocess_uint16)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds, n_train, val_ds, n_val, n_labels = prepare_dataset(df_excel, label_col, train_bool_col, img_as_uint16=True)
    train_steps = math.ceil(n_train/batch_size)
    val_steps = math.ceil(n_val/batch_size)

    train_truc_ds = train_ds.map(trunc_name) #get rid of name/idx
    val_trunc_ds = val_ds.map(trunc_name) #get rid of name/idx

    train_rot_ds = train_truc_ds.cache().map(augment)
    val_rot_ds = val_trunc_ds.cache().map(val_map)

    train_only_pos_aug_ds = train_truc_ds.map(position_augment)

    train_iter = iter(train_ds)
    train_rot_iter = iter(train_rot_ds)
    train_pos_iter = iter(train_only_pos_aug_ds)

    train_item = next(train_iter)
    train_rot_item = next(train_rot_iter)
    train_pos_item = next(train_pos_iter)

    train_img, train_label, train_index = train_item
    utils.array_min_max(train_img)
    print(train_label.numpy(), train_index.numpy())

    rot_img, rot_label = train_rot_item
    utils.array_min_max(rot_img)
    print(rot_label.numpy(), rot_label.numpy()*180)

    pos_img, pos_label = train_pos_item
    utils.array_min_max(pos_img)

    plt.imshow(skimage.img_as_ubyte(np.uint16(rot_img)), 'gray')
    plt.imshow(skimage.img_as_ubyte(train_img), 'gray')

    val_iter = iter(val_rot_ds)
    val_item = next(val_iter)
