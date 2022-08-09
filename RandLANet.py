from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')

        with tf.variable_scope('layers'):
            # 由于 self.inference 定义的返回值由原来的 f_out 变为 f_out=[f_out1, f_out2, f_out3, f_out4, f_out5]，
            # 因此 self.logits 也改为 self.logitss，即多个特征图对应的预测标签列表，实际上是 f_out1,...,f_out5 里 5 个特征图 f1,1, ..., f5,5 对应的预测标签列表。
            self.logitss = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
        # with tf.variable_scope('sub_loss1'):
            # 取出 self.logitss 中利用 f_out1 预测出来的标签来计算子损失 sub_loss1
            self.logits = tf.reshape(self.logitss[0], [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss1 = self.get_loss(valid_logits, valid_labels, self.class_weights)

        # with tf.variable_scope('sub_loss2'):
            # 取出 self.logitss 中利用 f_out2 预测出来的标签来计算子损失 sub_loss2
            self.logits = tf.reshape(self.logitss[1], [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss2 = self.get_loss(valid_logits, valid_labels, self.class_weights)

        # with tf.variable_scope('sub_loss3'):
            # 取出 self.logitss 中利用 f_out3 预测出来的标签来计算子损失 sub_loss3
            self.logits = tf.reshape(self.logitss[2], [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss3 = self.get_loss(valid_logits, valid_labels, self.class_weights)

        # with tf.variable_scope('sub_loss4'):
            # 取出 self.logitss 中利用 f_out4 预测出来的标签来计算子损失 sub_loss4
            self.logits = tf.reshape(self.logitss[3], [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss4 = self.get_loss(valid_logits, valid_labels, self.class_weights)

        # with tf.variable_scope('sub_loss5'):
            # 取出 self.logitss 中利用 f_out5 预测出来的标签来计算子损失 sub_loss5
            self.logits = tf.reshape(self.logitss[4], [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss5 = self.get_loss(valid_logits, valid_labels, self.class_weights)

            # 此时 self.loss 为 5 个子损失的平均值，不再是单独由 f_out5 预测出来的损失。
            self.loss = (self.loss1 + self.loss2 + self.loss3 + self.loss4 + self.loss5)/5
        ####    tp  added<<<    ####

        # with tf.variable_scope('loss'):
        #     self.logits = tf.reshape(self.logits, [-1, config.num_classes])
        #     self.labels = tf.reshape(self.labels, [-1])

        #     # Boolean mask of points that should be ignored
        #     ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
        #     for ign_label in self.config.ignored_label_inds:
        #         ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

        #     # Collect logits and labels that are not ignored
        #     valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        #     valid_logits = tf.gather(self.logits, valid_idx, axis=0)
        #     valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

        #     # Reduce label values in the range of logit shape
        #     reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
        #     inserted_value = tf.zeros((1,), dtype=tf.int32)
        #     for ign_label in self.config.ignored_label_inds:
        #         reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        #     valid_labels = tf.gather(reducing_list, valid_labels_init)

        #     self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

            ####    tp添加的特征融合模块 start    ####
            if j == 4:
                j=4
                # f_interp_i_11 是由 f1,0 上采样获得的中间特征
                f_interp_i_11 = self.nearest_interpolation(f_encoder_list[-j-1], inputs['interp_idx'][-j-1])
                # f_decoder_i_11 是 f1,1 的特征图，由 f0,0 和 f_interp_i_11 特征拼接、卷积而得到
                f_decoder_i_11 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_interp_i_11], axis=3), 32, [1, 1],
                                        'decoder_11',
                                        [1, 1], 'VALID', True, is_training)
                j=3
                # f_interp_i_21 是由 f2,0 上采样获得的中间特征
                f_interp_i_21 = self.nearest_interpolation(f_encoder_list[-j-1], inputs['interp_idx'][-j-1])
                # f_decoder_i_21 是 f2,1 的特征图，由 f1,0 和 f_interp_i_21 特征拼接、卷积而得到
                f_decoder_i_21 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_interp_i_21], axis=3), 32, [1, 1],
                                        'decoder_21',
                                        [1, 1], 'VALID', True, is_training)
                j=4
                # f_interp_i_22 是由 f2,1 上采样获得的中间特征
                f_interp_i_22 = self.nearest_interpolation(f_decoder_i_21, inputs['interp_idx'][-j-1])
                # f_decoder_i_22 是 f2,2 的特征图，由 f0,0、f1,1 和 f_interp_i_22 特征拼接、卷积而得到
                f_decoder_i_22 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_11, f_interp_i_22], axis=3), 32, [1, 1],
                                        'decoder_22',
                                        [1, 1], 'VALID', True, is_training)
                j=2
                # f_interp_i_31 是由 f3,0 上采样获得的中间特征
                f_interp_i_31 = self.nearest_interpolation(f_encoder_list[-j-1], inputs['interp_idx'][-j-1])
                # f_decoder_i_31 是 f3,1 的特征图，由 f2,0 和 f_interp_i_31 特征拼接、卷积而得到
                f_decoder_i_31 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_interp_i_31], axis=3), 128, [1, 1],
                                        'decoder_31',
                                        [1, 1], 'VALID', True, is_training)
                j=3
                # f_interp_i_32 是由 f3,1 上采样获得的中间特征
                f_interp_i_32 = self.nearest_interpolation(f_decoder_i_31, inputs['interp_idx'][-j-1])
                # f_decoder_i_32 是 f3,2 的特征图，由 f1,0、f2,1 和 f_interp_i_32 特征拼接、卷积而得到
                f_decoder_i_32 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_21, f_interp_i_32], axis=3), 32, [1, 1],
                                        'decoder_32',
                                        [1, 1], 'VALID', True, is_training)
                j=4
                # f_interp_i_33 是由 f3,2 上采样获得的中间特征
                f_interp_i_33 = self.nearest_interpolation(f_decoder_i_32, inputs['interp_idx'][-j-1])
                # f_decoder_i_33 是 f3,3 的特征图，由 f0,0、f1,1、f2,2 和 f_interp_i_33 特征拼接、卷积而得到
                f_decoder_i_33 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_11, f_decoder_i_22, f_interp_i_33], axis=3), 32, [1, 1],
                                        'decoder_33',
                                        [1, 1], 'VALID', True, is_training)
                j=1
                # f_interp_i_41 是由 f4,0 上采样获得的中间特征
                f_interp_i_41 = self.nearest_interpolation(f_encoder_list[-j-1], inputs['interp_idx'][-j-1])
                # f_decoder_i_41 是 f4,1 的特征图，由 f3,0 和 f_interp_i_41 特征拼接、卷积而得到
                f_decoder_i_41 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_interp_i_41], axis=3), 256, [1, 1],
                                        'decoder_41',
                                        [1, 1], 'VALID', True, is_training)
                j=2
                # f_interp_i_42 是由 f4,1 上采样获得的中间特征
                f_interp_i_42 = self.nearest_interpolation(f_decoder_i_41, inputs['interp_idx'][-j-1])
                # f_decoder_i_42 是 f4,2 的特征图，由 f2,0、f3,1 和 f_interp_i_42 特征拼接、卷积而得到
                f_decoder_i_42 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_31, f_interp_i_42], axis=3), 128, [1, 1],
                                        'decoder_42',
                                        [1, 1], 'VALID', True, is_training)
                j=3
                # f_interp_i_43 是由 f4,2 上采样获得的中间特征
                f_interp_i_43 = self.nearest_interpolation(f_decoder_i_42, inputs['interp_idx'][-j-1])
                # f_decoder_i_43、f1,0、f2,1、f3,2 和 f_interp_i_43 特征拼接、卷积而得到
                f_decoder_i_43 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_21, f_decoder_i_32, f_interp_i_43], axis=3), 32, [1, 1],
                                        'decoder_43',
                                        [1, 1], 'VALID', True, is_training)
                j=4
                # f_interp_i_44 是由 f4,3 上采样获得的中间特征
                f_interp_i_44 = self.nearest_interpolation(f_decoder_i_43, inputs['interp_idx'][-j-1])
                # f_decoder_i_44 是 f4,4 的特征图，由 f0,0、f1,1、f2,2、f3,3 和 f_interp_i_44 特征拼接、卷积而得到
                f_decoder_i_44 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_11, f_decoder_i_22, f_decoder_i_33, f_interp_i_44], axis=3), 32, [1, 1],
                                        'decoder_44',
                                        [1, 1], 'VALID', True, is_training)
                j=1
                # 由于该网络是在原来的特征图基础上进行融合，所以应保持原特征图的特征，不再自己提取f_interp_i_32
                # f_decoder_i_52 是 f5,2 的特征图，由 f3,0、f4,1 和 f_decoder_list[1] 特征拼接、卷积而得到
                f_decoder_i_52 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_41, f_decoder_list[1]], axis=3), 256, [1, 1],
                                        'decoder_52',
                                        [1, 1], 'VALID', True, is_training)
                j=2
                # f_decoder_i_53 是 f5,3 的特征图，由 f2,0、f3,1、f4,2 和 f_decoder_list[2] 特征拼接、卷积而得到
                f_decoder_i_53 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_31, f_decoder_i_42, f_decoder_list[2]], axis=3), 128, [1, 1],
                                        'decoder_53',
                                        [1, 1], 'VALID', True, is_training)
                j=3
                # f_decoder_i_54 是 f5,4 的特征图，由 f1,0、f2,1、f3,2、f4,3 和 f_decoder_list[3] 特征拼接、卷积而得到
                f_decoder_i_54 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_21, f_decoder_i_32, f_decoder_i_43, f_decoder_list[3]], axis=3), 32, [1, 1],
                                        'decoder_54',
                                        [1, 1], 'VALID', True, is_training)
                j=4
                # f_decoder_i_55 是 f5,5 的特征图，由 f0,0、f1,1、f2,2、f3,3、f4,4 和 f_decoder_list[4] 特征拼接、卷积而得到
                f_decoder_i_55 = helper_tf_util.conv2d(tf.concat([f_encoder_list[-j-2], f_decoder_i_11, f_decoder_i_22, f_decoder_i_33, f_decoder_i_44, f_decoder_list[4]], axis=3), 32, [1, 1],
                                        'decoder_55',
                                        [1, 1], 'VALID', True, is_training)
        # 将 f1,1 (f_decoder_i_11) 连接全连接层进行推理，得到推理结果 f_out1
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_i_11 , 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
            ####    tp添加的特征融合模块 over    ####
        # ###########################Decoder############################

        # f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out1 = tf.squeeze(f_layer_fc3, [2])   # tp

        ####    tp add >>>>    ####
        # 将 f2,2 (f_decoder_i_22) 连接全连接层进行推理，得到推理结果 f_out2
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_i_22 , 64, [1, 1], 'fc1-1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2-1', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc-1', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out2 = tf.squeeze(f_layer_fc3, [2])

        # 将 f3,3 (f_decoder_i_33) 连接全连接层进行推理，得到推理结果 f_out3
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_i_33 , 64, [1, 1], 'fc1-2', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2-2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc-2', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out3 = tf.squeeze(f_layer_fc3, [2])

        # 将 f4,4 (f_decoder_i_44) 连接全连接层进行推理，得到推理结果 f_out4
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_i_44 , 64, [1, 1], 'fc1-3', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2-3', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc-3', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out4 = tf.squeeze(f_layer_fc3, [2])

        # 将 f5,5 (f_decoder_i_55) 连接全连接层进行推理，得到推理结果 f_out5
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_i_55 , 64, [1, 1], 'fc1-4', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2-4', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc-4', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out5 = tf.squeeze(f_layer_fc3, [2])
        # f_out 包含 f1,1 到 f5,5 连接全连接层推理的结果 
        f_out=[f_out1, f_out2, f_out3, f_out4, f_out5]
        # f_out = tf.squeeze(f_layer_fc3, [2])  # original
        ####    tp add <<<<   ####

        return f_out

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
