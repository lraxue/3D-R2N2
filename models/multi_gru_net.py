# -*- coding: utf-8 -*-
# @Time    : 17-11-8 下午3:25
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : multi_gru_net.py
# @Software: PyCharm Community Edition

import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import Net, tensor5
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, \
    EltwiseMultiplyLayer, get_trainable_params, ConcatLayer


class MultiResGRUNet(Net):
    def network_definition(self):
        # (views, batch_size, 3, img_h, img_w)
        self.x = tensor5()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h
        n_gru_vox = [4, 8, 16, 32]

        n_convfilter = [8, 16, 32, 64, 128]
        n_fc_filters = [256]
        n_deconvfilter = [128, 64, 32, 16, 2]
        input_shape = (self.batch_size, 3, img_w, img_h)
        fc_shape = (self.batch_size, n_fc_filters[0])

        # To define the weights, define the net structure first
        x = InputLayer(input_shape)
        conv1a = ConvLayer(x, (n_convfilter[0], 7, 7))
        conv1b = ConvLayer(conv1a, (n_convfilter[0], 3, 3))
        pool1 = PoolLayer(conv1b)  # H/2

        conv2a = ConvLayer(pool1, (n_convfilter[1], 3, 3))
        conv2b = ConvLayer(conv2a, (n_convfilter[1], 3, 3))
        conv2c = ConvLayer(pool1, (n_convfilter[1], 1, 1))
        pool2 = PoolLayer(conv2c)  # H/4

        conv3a = ConvLayer(pool2, (n_convfilter[2], 3, 3))
        conv3b = ConvLayer(conv3a, (n_convfilter[2], 3, 3))
        conv3c = ConvLayer(pool2, (n_convfilter[2], 1, 1))
        pool3 = PoolLayer(conv3c)  # H/8

        conv4a = ConvLayer(pool3, (n_convfilter[3], 3, 3))
        conv4b = ConvLayer(conv4a, (n_convfilter[3], 3, 3))
        pool4 = PoolLayer(conv4b)  # H/16

        conv5a = ConvLayer(pool4, (n_convfilter[4], 3, 3))
        conv5b = ConvLayer(conv5a, (n_convfilter[4], 3, 3))
        conv5c = ConvLayer(pool4, (n_convfilter[4], 1, 1))  # H/32
        pool5 = PoolLayer(conv5b)

        flat5 = FlattenLayer(pool5)
        fc5 = TensorProductLayer(flat5, n_fc_filters[0])

        flat4 = FlattenLayer(pool4)
        fc4 = TensorProductLayer(flat4, n_fc_filters[0])

        flat3 = FlattenLayer(pool3)
        fc3 = TensorProductLayer(flat3, n_fc_filters[0])

        flat2 = FlattenLayer(pool2)
        fc2 = TensorProductLayer(flat2, n_fc_filters[0])

        # flat1 = FlattenLayer(pool1)
        # fc1 = TensorProductLayer(flat1, n_fc_filters[0])

        # ==================== recurrence 5 ========================#
        s_shape_5 = (self.batch_size, n_gru_vox[0], n_deconvfilter[0], n_gru_vox[0], n_gru_vox[0])
        # s_shape_5 = (self.batch_size, n_gru_vox[4], n_deconvfilter[4], n_gru_vox[4], n_gru_vox[4])
        prev_s_5 = InputLayer(s_shape_5)

        t_x_s_update_5 = FCConv3DLayer(prev_s_5, fc5, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))
        t_x_s_reset_5 = FCConv3DLayer(prev_s_5, fc5, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))

        reset_gate_5 = SigmoidLayer(t_x_s_reset_5)
        rs_5 = EltwiseMultiplyLayer(reset_gate_5, prev_s_5)
        t_x_rs_5 = FCConv3DLayer(rs_5, fc5, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))

        # ==================== recurrence 4 ========================#
        s_shape_4 = (self.batch_size, n_gru_vox[1], n_deconvfilter[1], n_gru_vox[1], n_gru_vox[1])
        prev_s_4 = InputLayer(s_shape_4)

        t_x_s_update_4 = FCConv3DLayer(prev_s_4, fc4, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3))
        t_x_s_reset_4 = FCConv3DLayer(prev_s_4, fc4, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3))

        reset_gate_4 = SigmoidLayer(t_x_s_reset_4)
        rs_4 = EltwiseMultiplyLayer(reset_gate_4, prev_s_4)
        t_x_rs_4 = FCConv3DLayer(rs_4, fc4, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3) )

        # =================== recurrence 3 =======================#
        s_shape_3 = (self.batch_size, n_gru_vox[2], n_deconvfilter[2], n_gru_vox[2], n_gru_vox[2])
        prev_s_3 = InputLayer(s_shape_3)

        t_x_s_update_3 = FCConv3DLayer(prev_s_3, fc3, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3))
        t_x_s_reset_3 = FCConv3DLayer(prev_s_3, fc3, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3))

        reset_gate_3 = SigmoidLayer(t_x_s_reset_3)
        rs_3 = EltwiseMultiplyLayer(reset_gate_3, prev_s_3)
        t_x_rs_3 = FCConv3DLayer(rs_3, fc3, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3))

        # ================== recurrence 2 =======================#
        s_shape_2 = (self.batch_size, n_gru_vox[3], n_deconvfilter[3], n_gru_vox[3], n_gru_vox[3])
        prev_s_2 = InputLayer(s_shape_2)

        t_x_s_update_2 = FCConv3DLayer(prev_s_2, fc2, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3))
        t_x_s_reset_2 = FCConv3DLayer(prev_s_2, fc2, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3))

        reset_gate_2 = SigmoidLayer(t_x_s_reset_2)
        rs_2 = EltwiseMultiplyLayer(reset_gate_2, prev_s_2)
        t_x_rs_2 = FCConv3DLayer(rs_2, fc2, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3))

        # # ================= recurrence 1 ========================#
        # s_shape_1 = (self.batch_size, n_gru_vox[4], n_deconvfilter[4], n_gru_vox[4], n_gru_vox[4])
        # prev_s_1 = InputLayer(s_shape_1)
        #
        # t_x_s_update_1 = FCConv3DLayer(prev_s_1, fc1, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3))
        # t_x_s_reset_1 = FCConv3DLayer(prev_s_1, fc1, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3))
        #
        # reset_gate_1 = SigmoidLayer(t_x_s_reset_1)
        # rs_1 = EltwiseMultiplyLayer(reset_gate_1, prev_s_1)
        # t_x_rs_1 = FCConv3DLayer(rs_1, fc1, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3))

        def encode_recurrence(x_curr):
            input_ = InputLayer(input_shape, x_curr)
            conv1a_ = ConvLayer(input_, (n_convfilter[0], 7, 7), params=conv1a.params)
            rect1a_ = LeakyReLU(conv1a_)
            conv1b_ = ConvLayer(rect1a_, (n_convfilter[0], 3, 3), params=conv1b.params)
            rect1_ = LeakyReLU(conv1b_)
            pool1_ = PoolLayer(rect1_)

            # flat1_ = FlattenLayer(pool1_)
            # fc1_ = TensorProductLayer(flat1_, n_fc_filters[0], params=fc1.params)
            # out1_ = LeakyReLU(fc1_)

            conv2a_ = ConvLayer(pool1_, (n_convfilter[1], 3, 3), params=conv2a.params)
            rect2a_ = LeakyReLU(conv2a_)
            conv2b_ = ConvLayer(rect2a_, (n_convfilter[1], 3, 3), params=conv2b.params)
            rect2_ = LeakyReLU(conv2b_)
            conv2c_ = ConvLayer(pool1_, (n_convfilter[1], 1, 1), params=conv2c.params)
            res2_ = AddLayer(conv2c_, rect2_)
            pool2_ = PoolLayer(res2_)

            flat2_ = FlattenLayer(pool2_)
            fc2_ = TensorProductLayer(flat2_, n_fc_filters[0], params=fc2.params)
            out2_ = LeakyReLU(fc2_)

            conv3a_ = ConvLayer(pool2_, (n_convfilter[2], 3, 3), params=conv3a.params)
            rect3a_ = LeakyReLU(conv3a_)
            conv3b_ = ConvLayer(rect3a_, (n_convfilter[2], 3, 3), params=conv3b.params)
            rect3_ = LeakyReLU(conv3b_)
            conv3c_ = ConvLayer(pool2_, (n_convfilter[2], 1, 1), params=conv3c.params)
            res3_ = AddLayer(conv3c_, rect3_)
            pool3_ = PoolLayer(res3_)

            flat3_ = FlattenLayer(pool3_)
            fc3_ = TensorProductLayer(flat3_, n_fc_filters[0], params=fc3.params)
            out3_ = LeakyReLU(fc3_)

            conv4a_ = ConvLayer(pool3_, (n_convfilter[3], 3, 3), params=conv4a.params)
            rect4a_ = LeakyReLU(conv4a_)
            conv4b_ = ConvLayer(rect4a_, (n_convfilter[3], 3, 3), params=conv4b.params)
            rect4_ = LeakyReLU(conv4b_)
            pool4_ = PoolLayer(rect4_)

            flat4_ = FlattenLayer(pool4_)
            fc4_ = TensorProductLayer(flat4_, n_fc_filters[0], params=fc4.params)
            out4_ = LeakyReLU(fc4_)

            conv5a_ = ConvLayer(pool4_, (n_convfilter[4], 3, 3), params=conv5a.params)
            rect5a_ = LeakyReLU(conv5a_)
            conv5b_ = ConvLayer(rect5a_, (n_convfilter[4], 3, 3), params=conv5b.params)
            rect5_ = LeakyReLU(conv5b_)
            conv5c_ = ConvLayer(pool4_, (n_convfilter[4], 1, 1), params=conv5c.params)
            res5_ = AddLayer(conv5c_, rect5_)
            pool5_ = PoolLayer(res5_)

            flat5_ = FlattenLayer(pool5_)
            fc5_ = TensorProductLayer(flat5_, n_fc_filters[0], params=fc5.params)
            out5_ = LeakyReLU(fc5_)

            return out5_.output, out4_.output, out3_.output, out2_.output  # , out1_.output

        s_encoder, _ = theano.scan(encode_recurrence,
                                   sequences=[self.x])
        out_5 = s_encoder[0]
        out_4 = s_encoder[1]
        out_3 = s_encoder[2]
        out_2 = s_encoder[3]
        # out_1 = s_encoder[4]

        def decode_recurrence_5(x_curr, prev_s_tensor, prev_in_gate_tensor):
            x_curr_ = InputLayer(fc_shape, x_curr)
            prev_s_5_ = InputLayer(s_shape_5, prev_s_tensor)
            t_x_s_update_5_ = FCConv3DLayer(prev_s_5_,
                                            x_curr_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                                            params=t_x_s_update_5.params)

            t_x_s_reset_5_ = FCConv3DLayer(prev_s_5_, x_curr_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                                           params=t_x_s_reset_5.params)

            update_gate_ = SigmoidLayer(t_x_s_update_5_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_5_)

            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_5_)
            t_x_rs_5_ = FCConv3DLayer(rs_, x_curr_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                                      params=t_x_rs_5.params)
            tanh_t_x_rs_ = TanhLayer(t_x_rs_5_)

            gru_out_5_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_5_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))

            return gru_out_5_.output, update_gate_.output

        s_update_5_, _ = theano.scan(decode_recurrence_5,
                                     sequences=[out_5],
                                     outputs_info=[tensor.zeros_like(np.zeros(s_shape_5),
                                                                     dtype=theano.config.floatX),
                                                   tensor.zeros_like(np.zeros(s_shape_5),
                                                                     dtype=theano.config.floatX)])
        update_all_5 = s_update_5_[-1]
        s_out_5 = update_all_5[0][-1]
        input_5 = InputLayer(s_shape_5, s_out_5)
        # Unpooling s_out_5
        unpool5 = Unpool3DLayer(input_5)
        conv_out5 = Conv3DLayer(unpool5, (64, 3, 3, 3))

        print("conv_out5", conv_out5.output_shape)

        def decode_recurrence_4(x_curr, prev_s_tensor, prev_in_gate_tensor):

            x_curr_ = InputLayer(fc_shape, x_curr)
            prev_s_4_ = InputLayer(s_shape_4, prev_s_tensor)
            t_x_s_update_4_ = FCConv3DLayer(prev_s_4_,
                                            x_curr_, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3),
                                            params=t_x_s_update_4.params)

            t_x_s_reset_4_ = FCConv3DLayer(prev_s_4_, x_curr_, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3),
                                           params=t_x_s_reset_4.params)

            print("x_curr: ", x_curr_.output_shape)
            print("prev_s_4_: ", prev_s_4_.output_shape)
            print("t_x_s_update_4_: ", t_x_s_update_4_.output_shape)
            print("t_x_s_reset_4_: ", t_x_s_reset_4_.output_shape)

            update_gate_ = SigmoidLayer(t_x_s_update_4_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_4_)

            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_4_)
            t_x_rs_4_ = FCConv3DLayer(rs_, x_curr_, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3),
                                      params=t_x_rs_4.params)
            tanh_t_x_rs_ = TanhLayer(t_x_rs_4_)

            gru_out_4_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_4_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))

            return gru_out_4_.output, update_gate_.output

        s_update_4_, _ = theano.scan(decode_recurrence_4,
                                     sequences=[out_4],
                                     outputs_info=[conv_out5.output,
                                                   tensor.zeros_like(np.zeros(s_shape_4),
                                                                     dtype=theano.config.floatX)])
        update_all_4 = s_update_4_[-1]
        s_out_4 = update_all_4[0][-1]
        input_4 = InputLayer(s_shape_4, s_out_4)
        # Unpooling s_out_4
        unpool4 = Unpool3DLayer(input_4)
        conv_out4 = Conv3DLayer(unpool4, (n_deconvfilter[2], 3, 3, 3))

        print("conv_out_4: ", conv_out4.output_shape)
        print("conv_out_4: ", conv_out4.output)

        def decode_recurrence_3(x_curr, prev_s_tensor, prev_in_gate_tensor):
            x_curr_ = InputLayer(fc_shape, x_curr)
            prev_s_3_ = InputLayer(s_shape_3, prev_s_tensor)
            t_x_s_update_3_ = FCConv3DLayer(prev_s_3_,
                                            x_curr_, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3),
                                            params=t_x_s_update_3.params)

            t_x_s_reset_3_ = FCConv3DLayer(prev_s_3_, x_curr_, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3),
                                           params=t_x_s_reset_3.params)

            update_gate_ = SigmoidLayer(t_x_s_update_3_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_3_)

            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_3_)
            t_x_rs_3_ = FCConv3DLayer(rs_, x_curr_, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3),
                                      params=t_x_rs_3.params)
            tanh_t_x_rs_ = TanhLayer(t_x_rs_3_)

            gru_out_3_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_3_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))

            return gru_out_3_.output, update_gate_.output

        s_update_3_, _ = theano.scan(decode_recurrence_3,
                                     sequences=[out_3],
                                     outputs_info=[conv_out4.output,
                                                   tensor.zeros_like(np.zeros(s_shape_3),
                                                                     dtype=theano.config.floatX)])
        update_all_3 = s_update_3_[-1]
        s_out_3 = update_all_3[0][-1]
        input_3 = InputLayer(s_shape_3, s_out_3)
        # Unpooling s_out_4
        unpool3 = Unpool3DLayer(input_3)
        conv_out3 = Conv3DLayer(unpool3, (n_deconvfilter[3], 3, 3, 3))

        print("conv_out_3: ", conv_out3.output_shape)
        print("conv_out_3: ", conv_out3.output)

        def decode_recurrence_2(x_curr, prev_s_tensor, prev_in_gate_tensor):
            x_curr_ = InputLayer(fc_shape, x_curr)
            prev_s_2_ = InputLayer(s_shape_2, prev_s_tensor)
            t_x_s_update_2_ = FCConv3DLayer(prev_s_2_,
                                            x_curr_, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3),
                                            params=t_x_s_update_2.params)

            t_x_s_reset_2_ = FCConv3DLayer(prev_s_2_, x_curr_, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3),
                                           params=t_x_s_reset_2.params)

            update_gate_ = SigmoidLayer(t_x_s_update_2_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_2_)

            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_2_)
            t_x_rs_2_ = FCConv3DLayer(rs_, x_curr_, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3),
                                      params=t_x_rs_2.params)
            tanh_t_x_rs_ = TanhLayer(t_x_rs_2_)

            gru_out_2_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_2_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))

            return gru_out_2_.output, update_gate_.output

        s_update_2_, _ = theano.scan(decode_recurrence_2,
                                     sequences=[out_2],
                                     outputs_info=[conv_out3.output,
                                                   tensor.zeros_like(np.zeros(s_shape_2),
                                                                     dtype=theano.config.floatX)])
        update_all_2 = s_update_2_[-1]
        s_out_2 = update_all_2[0][-1]
        input_2 = InputLayer(s_shape_2, s_out_2)
        # Unpooling s_out_4
        # unpool2 = Unpool3DLayer(input_2)
        # conv_out2 = Unpool3DLayer(unpool2, (n_deconvfilter[4], 3, 3, 3))

        # def decode_recurrence_1(x_curr, prev_s_tensor, prev_in_gate_tensor):
        #     x_curr_ = InputLayer(fc_shape, x_curr)
        #     prev_s_1_ = InputLayer(s_shape_1, prev_s_tensor)
        #     t_x_s_update_1_ = FCConv3DLayer(prev_s_1_,
        #                                     x_curr_, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3),
        #                                     params=t_x_s_update_1.params)
        #
        #     t_x_s_reset_1_ = FCConv3DLayer(prev_s_1_, x_curr_, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3),
        #                                    params=t_x_s_reset_1.params)
        #
        #     update_gate_ = SigmoidLayer(t_x_s_update_1_)
        #     comp_update_gate_ = ComplementLayer(update_gate_)
        #     reset_gate_ = SigmoidLayer(t_x_s_reset_1_)
        #
        #     rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_1_)
        #     t_x_rs_1_ = FCConv3DLayer(rs_, x_curr_, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3),
        #                               params=t_x_rs_1.params)
        #     tanh_t_x_rs_ = TanhLayer(t_x_rs_1_)
        #
        #     gru_out_1_ = AddLayer(
        #         EltwiseMultiplyLayer(update_gate_, prev_s_1_),
        #         EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))
        #
        #     return gru_out_1_.output, update_gate_.output
        #
        # s_update_1_, _ = theano.scan(decode_recurrence_1,
        #                              sequences=[out_1],
        #                              outputs_info=[conv_out2.output,
        #                                            tensor.zeros_like(np.zeros(s_shape_1),
        #                                                              dtype=theano.config.floatX)])
        # update_all_1 = s_update_1_[-1]
        # s_out_1 = update_all_1[0][-1]
        #
        # s_out_1_input = InputLayer(s_shape_1, s_out_1)
        conv_out2 = Conv3DLayer(input_2, (n_deconvfilter[4], 3, 3, 3))
        softmax_loss = SoftmaxWithLoss3D(conv_out2.output)


        print("conv_out_2: ", conv_out2.output_shape)
        print("conv_out_2: ", conv_out2.output)

        self.loss = softmax_loss.loss(self.y)
        self.error = softmax_loss.error(self.y)
        self.params = get_trainable_params()
        self.output = softmax_loss.prediction()
        self.activations = [update_all_5, update_all_4, update_all_3, update_all_2]
        # self.activations = [update_all_5, update_all_4, update_all_3, update_all_2, update_all_1]
