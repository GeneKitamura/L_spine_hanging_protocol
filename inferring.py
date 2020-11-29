import pandas as pd
import numpy as np
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt

from skimage import exposure, transform

import sys
sys.path.append('..')

import XR_ML_tools.model_tools as model_tools
from XR_ML_tools import utils
from XR_ML_tools.fetch_data import prepare_dataset
from XR_ML_tools import image_tools

train_numpy_keras = model_tools.train_numpy_keras
rotate_augment = model_tools.rotate_augment
val_rot_map = model_tools.val_rot_map
preprocess_densenet = model_tools.preprocess_densenet
trunc_name = model_tools.trunc_name
load_densenet = model_tools.load_densenet

iterate_tile = image_tools.iterate_tile
tile_alt_imshow = image_tools.tile_alt_imshow

def infer_regression():
    rand_rot_angle = 100
    model_weights = './rot/model_3495_100deg'
    out_excel = './rot/rot_out_3495_100deg.xlsx'
    # model_weights = './rot/model_3495_120deg'
    # out_excel = './rot/rot_out_3495_120deg.xlsx'

    label_col = 'angle'
    train_bool_col = 'new_train'
    batch_size = 20
    train_npz = './pos_hard_npz/238_0to3495.npz'
    val_npz = './pos_hard_npz/224_0to3495.npz'

    df_excel = './pos_hard_processed/rot_labeled_0to3945.xlsx'
    labels_excel = './pos_hard_processed/0to3945_newposlabel.xlsx'

    out_metric_final = pd.read_excel(out_excel)
    for i in ['loss']:
        plt.plot(out_metric_final.index, out_metric_final[i])
        plt.title(i)

    val_map = val_rot_map(rand_rot_angle)
    preprocess_map = preprocess_densenet
    preprocess_uint16 = True
    inner_preprocess = preprocess_map(uint16=preprocess_uint16)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    n_class = 1
    activation = tf.keras.activations.tanh

    train_ds, n_train, val_ds, n_val, test_ds, n_test, n_labels = prepare_dataset(df_excel, label_col, train_bool_col,
                                                                 img_as_uint16=True, flip_RtoL=True,
                                                                 npz_df=labels_excel, train_npz=val_npz,
                                                                 val_npz=val_npz)
    c_ds = val_ds
    orig_ds = c_ds.batch(500)
    processed_ds = c_ds.map(trunc_name).cache().map(val_map).map(inner_preprocess).batch(500)

    processed_iter = iter(processed_ds)
    orig_iter = iter(orig_ds)

    net = load_densenet(n_class=n_class, activation=activation)
    net.load_weights(model_weights)

    orig_item = next(orig_iter)
    print('\norig')
    orig_img, orig_label, orig_idx = orig_item
    utils.array_min_max(orig_img)

    processed_item = next(processed_iter)
    print('\nprocessed_item')
    proc_imgs, proc_labels = processed_item
    scaled_proc_labels = proc_labels * rand_rot_angle
    utils.array_min_max(proc_imgs)

    pred_angle = net.predict(proc_imgs)
    scaled_pred_angle = pred_angle * rand_rot_angle
    scaled_pred_angle = scaled_pred_angle.reshape([proc_imgs.shape[0]])

    print(proc_imgs.shape)
    utils.array_min_max(proc_labels)
    utils.array_min_max(pred_angle)

    diff = np.abs(scaled_proc_labels - scaled_pred_angle)
    print(np.mean(diff))

    iterate_tile(skimage.img_as_ubyte(orig_img), titles=list(zip(orig_idx.numpy(),
                                                                 np.int32(scaled_proc_labels),
                                                                 np.int32(scaled_pred_angle))))

    def show_orig_pred_true_images():
        img_list = []
        angle_list = []
        count = 0
        rot_img_ids = [20, 28, 44, 54, 58, 75, 99, 102, 104, 131, 157, 162, 190, 213, 218, 219, 275, 307, 309, 328]

        for i in rot_img_ids:
            img_id = i
            c_img = orig_img[img_id, ..., 0].numpy()
            pred_angle = scaled_pred_angle[img_id]
            true_angle = scaled_proc_labels[img_id].numpy()
            resize_it = False

            pred_rotate_img = exposure.equalize_hist(transform.rotate(c_img, pred_angle, resize=resize_it))
            true_rotate_img = exposure.equalize_hist(transform.rotate(c_img, true_angle, resize=resize_it))
            img_list += [c_img, true_rotate_img, pred_rotate_img]
            angle_list += [0, "true {}".format(int(true_angle)), "pred {}".format(int(pred_angle))]
            count += 1

        img_list = np.stack(img_list, axis=0)

        tile_alt_imshow(img_list, titles=angle_list, h_slot=count, w_slot=3, height=260)