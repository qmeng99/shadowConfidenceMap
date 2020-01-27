from __future__ import division
import os
import tensorflow as tf
import numpy as np
import random
import pdb
import residual_modef

height = 224
width = 288
batch_size = 25
l_r_f = 1e-3
model_dir = './model'
logs_path = './model'
max_iter_step = 90100


def read_decode(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={"Cdata": tf.FixedLenFeature([], tf.string),
                  "Clabel": tf.FixedLenFeature([], tf.int64)})
    Clsi_data = tf.decode_raw(features["Cdata"], tf.float32)
    Clsi_label = tf.cast(features["Clabel"], tf.int64)
    Clsi_data = tf.reshape(Clsi_data, [height, width, 1])
    Clsi_datas, Clsi_labels = tf.train.batch([Clsi_data, Clsi_label], batch_size=batch_size, capacity=1000, num_threads=8)
    # images and labels are tensor object
    return Clsi_datas, Clsi_labels

def read_dataforBrain(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
       features={"Bdata": tf.FixedLenFeature([], tf.string),
                 "Blabel": tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features["Bdata"], tf.float32)
    mask = tf.decode_raw(features["Blabel"], tf.float32)
    image = tf.reshape(image, [height, width, 1])
    mask = tf.reshape(mask, [height, width])
    images, masks = tf.train.batch([image, mask], batch_size=batch_size, capacity=1000, num_threads=8)  # output shape (10,240,240,1),(10,240,240,1)
    # images and labels are tensor object
    return images, masks

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def load():
    filename = "/home/train_comb_clssifier.tfrecords"
    filename_queue = tf.train.string_input_producer([filename])
    c_data, c_lbl = read_decode(filename_queue)

    filename_brain = '/home/Brain_train.tfrecords'
    filename_queue_brain = tf.train.string_input_producer([filename_brain])
    image_brain, mask_brain = read_dataforBrain(filename_queue_brain)

    return c_data, c_lbl, image_brain, mask_brain

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
    zeromask = tf.constant(0, shape=[batch_size, height, width, 1], dtype=tf.float32)
    maskpos = tf.where(tf.less_equal(dis_fn_all, zeromask), zeromask, dis_fn_all)  # all fn pixels that smaller than 100% sure shadow are 1 in the mask
    # get fn smaller part ratio
    r_small = 1 - ((tf.abs(maskpos)) / (mean_tp - min_v + e))
    conf_small = tf.where(tf.equal(maskpos, zeromask), zeromask, r_small)

    # distance<0 mask
    maskneg = tf.where(tf.greater_equal(dis_fn_all, zeromask), zeromask, dis_fn_all)  # all fn pixels that greater than 100% sure shadow are 1 in the mask
    # get fn greater part ratio
    r_big = 1 - (tf.abs(maskneg) / (max_v - mean_tp + e))
    conf_big = tf.where(tf.equal(maskneg, zeromask), zeromask, r_big)
    confidence = conf_small + conf_big + tp

    return confidence

def build_gpu():
    with tf.device("/gpu:0"):

        l_r_A = tf.Variable(l_r_f, dtype=tf.float32, trainable=False)
        opt_clsi = tf.train.MomentumOptimizer(learning_rate=l_r_A, momentum=0.9)
        opt_f = tf.train.MomentumOptimizer(learning_rate=l_r_A, momentum=0.9)
        C_data, C_lbl, B_data, B_lbl = load()

        # data augmentation: adding noise
        B_data = gaussian_noise_layer(B_data, 0.1)

        # set the image level label for the brain images.
        # All brain images are shadow images thus image-level labels should be all 1
        img_data_lbl = np.zeros(shape=(batch_size, 2))
        img_data_lbl[:, 1] = 1
        img_lbl = tf.constant(img_data_lbl, dtype=tf.int64)

        C_data_new = tf.expand_dims(C_data, 1)

        # --------------step 1 train classifer
        with tf.variable_scope('classifier'):
            x_1, res_scales_1, saved_strides_1, filters_1, logit_lbl = residual_modef.residual_encoder(inputs=C_data_new,
                num_classes=2,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64),
                strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # classification loss
        labels_onehot = tf.one_hot(C_lbl, depth=2)
        c_loss_all = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_lbl, labels=labels_onehot))

        B_data_new = tf.expand_dims(B_data, 1)

        # --------------step 2 train segmentation and finetune classification
        with tf.variable_scope('classifier', reuse=True):
            dec_x, dec_res_scales, dec_saved_strides, dec_filters, loglbl = residual_modef.residual_encoder(
                inputs=B_data_new,
                num_classes=2,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64),
                strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        c_loss_S = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=loglbl, labels=img_lbl))

        with tf.variable_scope('segmentation'):
            net_output_ops = residual_modef.residual_decoder(
                inputs=dec_x,
                num_classes=2,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=dec_filters,
                res_scales=dec_res_scales,
                saved_strides=dec_saved_strides,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            logitsyn = net_output_ops['logits']

        # shadow image classification loss
        B_lbl = tf.cast(B_lbl, tf.int32)
        B_lbl_new = tf.expand_dims(B_lbl, 1)
        seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logitsyn, labels=B_lbl_new))
        l2regloss_BiSeg = tf.losses.get_regularization_loss('segmentation')

        lossall = seg_loss + l2regloss_BiSeg

        softmax_logit = net_output_ops['y_']
        rec_logit = tf.cast(tf.expand_dims(softmax_logit, 4), tf.int32)

        # --------------step 3 train transfer Network, from output of transfer matrix to reference confidence map
        # get the confidence map for training data ground truth
        exp_gt = tf.expand_dims(B_lbl, 3)
        confidencetrain = Binary2Confidence(exp_gt, rec_logit[:, 0, :, :, :], B_data)

        confidencetrain_new = tf.expand_dims(confidencetrain, 1)

        seg_logit = tf.cast(rec_logit, tf.float32)

        with tf.variable_scope('transNet'):
            net_output_ops_T = residual_modef.residual_unet_3d(
                inputs=seg_logit,
                num_classes=1,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32),
                strides=((1, 1, 1), (1, 2, 2), (1, 2, 2)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            logittransNet = net_output_ops_T['logits']

        loss_transNet = tf.losses.mean_squared_error(logittransNet, confidencetrain_new)
        l2regloss_TransNet = tf.losses.get_regularization_loss('transNet')

        losstransNet_all = loss_transNet + l2regloss_TransNet

        # --------------step 4 train confidence unet, from input to reference confidence map

        with tf.variable_scope('confseg'):
            net_output_ops_C = residual_modef.residual_unet_3d(
                inputs=B_data_new,
                num_classes=1,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64),
                strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            logitconfseg = net_output_ops_C['logits']

        loss_confseg = tf.losses.mean_squared_error(logitconfseg, logittransNet)
        loss_confseg_TM = tf.losses.mean_squared_error(logitconfseg, confidencetrain_new)
        l2regloss_ConfSeg = tf.losses.get_regularization_loss('confseg')

        lossconfseg_all = loss_confseg + l2regloss_ConfSeg

        # ---------------optimization-------------------------------------------------------
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classifier')
        seg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'segmentation')
        confSeg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'confseg')
        transNet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'transNet')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            clsi_train_op = opt_clsi.minimize(c_loss_all, var_list=c_vars, colocate_gradients_with_ops=True)
            f_train_op = opt_f.minimize(lossall, var_list=[c_vars, seg_vars])
            tranNet_train_op = opt_f.minimize(losstransNet_all, var_list=transNet_vars)
            confseg_train_op = opt_f.minimize(lossconfseg_all, var_list=confSeg_vars)

        return clsi_train_op, f_train_op, tranNet_train_op, confseg_train_op, c_loss_all, c_loss_S, seg_loss, lossall, \
               loss_transNet, loss_confseg_TM, logittransNet, \
               B_data, B_lbl, rec_logit, \
               confidencetrain, logitconfseg, loss_confseg


def main():
    c_train_op, f_train_op, tranNet_train_op, confseg_train_op, c_loss_all, c_loss_S, seg_loss, lossall, \
    loss_transNet, loss_confseg_TM, logittransNet, \
    B_data, B_lbl_train, rec_logit, \
    confidencetrain, logitconfseg, loss_confseg = build_gpu()
    # ------------------------validation----------------------
    image = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
    label = tf.placeholder(dtype=tf.float32, shape=[None, height, width])

    image_new = tf.expand_dims(image, 1)
    with tf.variable_scope('classifier', reuse=True):
        dec_x, dec_res_scales, dec_saved_strides, dec_filters, loglbl = residual_modef.residual_encoder(
            inputs=image_new,
            num_classes=2,
            num_res_units=1,
            mode=tf.estimator.ModeKeys.EVAL,
            filters=(8, 16, 32, 64),
            strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
    with tf.variable_scope('segmentation', reuse=True):
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

    with tf.variable_scope('transNet', reuse=True):
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


    with tf.variable_scope('confseg', reuse=True):
        net_output_ops_C = residual_modef.residual_unet_3d(
            inputs=image_new,
            num_classes=1,
            num_res_units=1,
            mode=tf.estimator.ModeKeys.EVAL,
            filters=(8, 16, 32, 64),
            strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        logitconfS = net_output_ops_C['logits']

        loss_confS_val = tf.losses.mean_squared_error(logitconfS, logittransN)
        loss_confS_TM_val = tf.losses.mean_squared_error(logitconfS, confidenceVal_new)

        # loss_confS_val = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitconfS, labels=confidenceVal_new))

    # ----------------------------------------------------------
    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a summary to monitor cost tensor
        tf.summary.scalar("c_loss_all", c_loss_all)
        tf.summary.scalar("c_loss_S", c_loss_S)
        tf.summary.scalar("seg_loss", seg_loss)
        tf.summary.scalar("lossall", lossall)  # classification and segmentation combine loss Training
        tf.summary.scalar("loss_validation", loss_softce)   # segmentation loss Validation
        tf.summary.scalar("loss_transN", loss_transNet)  # transfer Network loss Training
        tf.summary.scalar("loss_transN_val", loss_transN_val)  # transfer Network loss Validation
        tf.summary.scalar("loss_confS", loss_confseg)  # confidence estimation loss Training
        tf.summary.scalar("loss_confS_val", loss_confS_val)# confidence estimation loss Validation
        tf.summary.scalar("loss_confseg_TM", loss_confseg_TM)# confidence estimation to Transfer Matrix loss Training
        tf.summary.scalar("loss_confS_TM_val", loss_confS_TM_val)# confidence estimation to Transfer Matrix loss Validation

        tf.summary.image('Orig', B_data)
        tf.summary.image('GT', tf.cast(tf.expand_dims(B_lbl_train, 3), tf.float32)) # segmentation ground truth Training
        tf.summary.image('seg', tf.cast(rec_logit[:,0,:,:,:], tf.float32)) # segmentation estimation Training
        tf.summary.image('segconf_asGT', confidencetrain)   # confidence map from Transfer Matrix Training
        tf.summary.image('SegConf', logitconfseg[:,0,:,:,:]) # confidence estimation from network learning Training
        tf.summary.image('TransConf', logittransNet[:,0,:,:,:]) # confidence map from netwrok learning Training


        tf.summary.image('val_B_GT', tf.cast(tf.expand_dims(B_lbl, 3), tf.float32))   # segmentation ground truth Val
        tf.summary.image('val_B_seg', tf.cast(exp_logit_B[:, 0, :, :, :], tf.float32))  # segmentation estimation Val
        tf.summary.image('segConfGT_val', confidenceVal)  # confidence map from Transfer Matrix Val
        tf.summary.image('segConf_val', logitconfS[:,0,:,:,:])  # confidence map from Transfer Matrix Val
        tf.summary.image('transConf_val', logittransN[:,0,:,:,:])  # confidence map from netwrok learning  Val
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        sess.run(init_op)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        testpath = '/home/Brain_val.npz'
        testimg = np.load(testpath)['imgtest']
        testgt = np.load(testpath)['segtest']

        lossce = 0
        loss_CS = 0
        loss_TN = 0
        loss_CS_TM = 0
        for i in range(max_iter_step):
            indexsel = random.sample(range(0, testimg.shape[0]), batch_size)
            t_data = np.reshape(testimg[indexsel, :, :], (batch_size, height, width, 1))
            t_seggt = np.reshape(testgt[indexsel, :, :], (batch_size, height, width))
            feed_dict = {image: t_data, label: t_seggt}

            if i < 2e4+10:  # train classification (about 20k)
                _, summary = sess.run([c_train_op, merged_summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, i)
            elif i < 5e4+10:  # finetune train segmentation, classifier (about 20k)  +10 make sure that 4e4 will be saved
                _, summary, lossce = sess.run([f_train_op, merged_summary_op, loss_softce], feed_dict=feed_dict)
                summary_writer.add_summary(summary, i)
            elif i < 6e4+10: # train transfer network (about 10k)
                _, summary, loss_TN = sess.run([tranNet_train_op, merged_summary_op, loss_transN_val], feed_dict=feed_dict)
                summary_writer.add_summary(summary, i)
            else:
                _, summary, loss_CS, loss_CS_TM = sess.run([confseg_train_op, merged_summary_op, loss_confS_val, loss_confS_TM_val], feed_dict=feed_dict)
                summary_writer.add_summary(summary, i)


            loss_classifer, loss_seg, losstransnet, lossconfseg, lossConfsegTM = sess.run([c_loss_S, seg_loss, loss_transNet, loss_confseg, loss_confseg_TM])

            if i % 100 == 0:
                print("i = %d" % i)
                print ("Train classification Loss = {}".format(loss_classifer))
                print ("Train segmentation Loss = {}".format(loss_seg))
                print ("Validation segmentation Loss = {}".format(lossce))
                print ("Train Conf_seg Loss = {}".format(lossconfseg))
                print ("Validation Confseg Loss = {}".format(loss_CS))
                print ("Train Trans_Net Loss = {}".format(losstransnet))
                print ("Validation TransNet Loss = {}".format(loss_TN))
                print ("Train confseg_TM Loss = {}".format(lossConfsegTM))
                print ("Validation confseg_TM Loss = {}".format(loss_CS_TM))

            if i % 500 == 0:
                saver.save(sess, os.path.join(model_dir, "model.val"), global_step=i)
        coord.request_stop()
        coord.join(threads)


main()