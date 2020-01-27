from __future__ import division
import os
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import random
import pdb
from scipy.misc import imsave, imread
import scipy.io as sio


import residual_modef
import measurement

height = 224
width = 288
threshold = 0.3
ckpt_dir = '/home/model'

def categorical_dice(pred, truth, k, eps=1e-9):
    # Dice overlap metric for label value k
    A = tf.cast(tf.equal(pred, k), dtype=tf.float32)
    B = tf.cast(tf.equal(truth, k), dtype=tf.float32)
    return 2 * tf.reduce_sum(tf.multiply(A, B)) / (tf.reduce_sum(A) + tf.reduce_sum(B) + eps)

def soft_dice(pred, truth, eps=1e-9):
    # Dice overlap metric for label value k
    A = tf.cast(pred, dtype=tf.float32)
    B = tf.cast(truth, dtype=tf.float32)
    return 2 * tf.reduce_sum(tf.multiply(A[:,:,:,0], B)) / (tf.reduce_sum(A) + tf.reduce_sum(B) + eps)

def Binary2Confidence(gtimg, predimg_int, origimg):
    e = tf.constant(1e-9)
    max_v = tf.reduce_max(origimg, axis=[1,2], keep_dims=True)  # [Batchsize, H, W, 1]
    min_v = tf.reduce_min(origimg, axis=[1,2], keep_dims=True)  # [Batchsize, H, W, 1]
    tp = tf.multiply(gtimg, predimg_int)  # [batchsize, H, W, 1]
    tp = tf.cast(tp, tf.float32)

    intensity_overlap = tf.multiply(tp, origimg)  # [batchsize, H, W, 1]
    mean_tp = tf.reduce_mean(intensity_overlap, axis=[1, 2], keep_dims=True)

    predimg = tf.cast(predimg_int, tf.float32)
    intensity_fn = tf.multiply((predimg - tp), origimg)
    # get intensity distrance from each fn to 100% sure shadows, distance>0 use min value to get ratio while distance<0 use max value to get ratio
    dis_all = mean_tp - intensity_fn # dis_all includes background, not only fn parts
    dis_fn_all = tf.multiply((predimg - tp), dis_all)

    # distance>0 mask
    zeromask = tf.constant(0, shape=[1, height, width, 1], dtype=tf.float32)
    maskpos = tf.where(tf.less_equal(dis_fn_all, zeromask), zeromask, dis_fn_all)  # all fn pixels that smaller than 100% sure shadow are 1 in the mask
    # get fn smaller part ratio
    r_small = 1 - ((tf.abs(maskpos)) / (mean_tp - min_v + e))
    conf_small = tf.where(tf.equal(maskpos, zeromask), zeromask, r_small)

    # distance<0 mask
    maskneg = tf.where(tf.greater_equal(dis_fn_all, zeromask), zeromask, dis_fn_all)  # all fn pixels that smaller than 100% sure shadow are 1 in the mask
    # get fn smaller part ratio
    r_big = 1 - (tf.abs(maskneg) / (max_v - mean_tp + e))
    conf_big = tf.where(tf.equal(maskneg, zeromask), zeromask, r_big)
    confidence = conf_small + conf_big + tp

    return confidence

def main():
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/gpu:0"):
            image = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
            label = tf.placeholder(dtype=tf.float32, shape=[None, height, width])

            image_new = tf.expand_dims(image, 1)
            with tf.variable_scope('classifier'):
                dec_x, dec_res_scales, dec_saved_strides, dec_filters, loglbl = residual_modef.residual_encoder(
                    inputs=image_new,
                    num_classes=2,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32, 64),
                    strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
            with tf.variable_scope('segmentation'):
                net_output_ops = residual_modef.residual_decoder(
                    inputs=dec_x,
                    num_classes=2,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=dec_filters,
                    res_scales=dec_res_scales,
                    saved_strides=dec_saved_strides,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

                logt2 = net_output_ops['logits']

            softmax_logit = net_output_ops['y_']
            exp_logit_B = tf.cast(tf.expand_dims(softmax_logit, 4), tf.int32)

            B_lbl = tf.cast(label, tf.int32)
            B_lbl_new = tf.expand_dims(B_lbl, 1)
            loss_softce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logt2, labels=B_lbl_new))

            exp_gt_B = tf.expand_dims(B_lbl, 3)
            confidenceVal = Binary2Confidence(exp_gt_B, exp_logit_B[:, 0, :, :, :], image)

            confidenceVal_new = tf.expand_dims(confidenceVal, 1)

            exp_seg_logit = tf.cast(exp_logit_B, tf.float32)

            with tf.variable_scope('transNet'):
                net_output_ops_T = residual_modef.residual_unet_3d(
                    inputs=exp_seg_logit,
                    num_classes=1,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32),
                    strides=((1, 1, 1), (1, 2, 2), (1, 2, 2)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

                logittransN = net_output_ops_T['logits']
                loss_transN_val = tf.losses.mean_squared_error(logittransN, confidenceVal_new)

            with tf.variable_scope('confseg'):
                net_output_ops_C = residual_modef.residual_unet_3d(
                    inputs=image_new,
                    num_classes=1,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32, 64),
                    strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

                logitconfS = net_output_ops_C['logits']

                loss_confS = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logitconfS, labels=confidenceVal_new))

            predic = exp_logit_B[0,0,:,:,0]
            diceloss = categorical_dice(predic, B_lbl, 1)
            softdice = soft_dice(logitconfS[:,0,:,:], confidenceVal[0,:,:,0])

            softdice_TM = soft_dice(logittransN[:, 0, :, :, :], confidenceVal[0, :, :, 0])

        # ---------------------------------------------------
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

            testpath = '/home/Brain_test.npz'
            testimg = np.load(testpath)['imgtest']
            testgt = np.load(testpath)['segtest']

            dice_all = []
            recallall = []
            precisionall = []
            softDSCall = []
            softDSCall_TM = []
            mse_all = []
            for i in range(testimg.shape[0]):
                t_data = np.reshape(testimg[i, :, :], (1, height, width, 1))
                t_seggt = np.reshape(testgt[i,:,:], (1, height, width))

                feed_dict = {image: t_data, label: t_seggt}
                BinarySeg, ConfSeg, Confinput, transNinput, dice, softD, softD_TM = sess.run(
                    [softmax_logit, logitconfS, confidenceVal, logittransN, diceloss, softdice, softdice_TM],
                    feed_dict=feed_dict)


                dice_all.append(dice)
                softDSCall.append(softD)
                softDSCall_TM.append(softD_TM)

                dgt, dpred = measurement.sepdice(t_seggt[0, :, :], BinarySeg[0, 0, :, :])


                recallall.append(dgt)
                precisionall.append(dpred)

                new_img = np.copy(ConfSeg[0, 0, :, :, 0])
                new_img[ConfSeg[0, 0, :, :, 0] < threshold] = 0

                floatseggt = t_seggt.astype(np.float32)

                mse_v = measurement.mse(BinarySeg[0, 0, :, :], floatseggt[0, :, :])
                mse_all.append(mse_v)

                print ((
                    'recall -- {}, precision -- {}, dice -- {}, softDSC -- {}, mse -- {}, softDSC_TM -- {}').format(
                    dgt, dpred,
                    dice, softD,
                    mse_v, softD_TM))

            print ((
                'Mean dice: {}, Mean recall: {}, Mean precision: {}, Mean softDSC: {}, Mean mse: {}, Mean softDSC_TM: {}').format(
                np.array(dice_all).mean(), np.array(recallall).mean(), np.array(precisionall).mean(),
                np.array(softDSCall).mean(), np.array(mse_all).mean(), np.array(softDSCall_TM).mean()))
            print ((
                'Std dice: {}, Std recall: {}, Std precision: {}, Std softDSC: {}, Std mse: {}, Std softDSC_TM: {}').format(
                np.array(dice_all).std(), np.array(recallall).std(), np.array(precisionall).std(),
                np.array(softDSCall).std(), np.array(mse_all).std(), np.array(softDSCall_TM).std()))


    return

main()