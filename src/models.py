import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import global_avg_pool
#######################
# 3d functions
#######################
# convolution


# 3D unet graph
def unet(inputI, output_channel):
    """3D U-net"""
    phase_flag = 1
    concat_dim = 4

    conv1_1 = conv3d(
        input=inputI,
        output_chn=64,
        kernel_size=3,
        stride=1,
        use_bias=False,
        name='conv1')
    # conv1_1 (1, 96, 96, 96, 64)
    conv1_bn = tf.contrib.layers.batch_norm(
        conv1_1,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=phase_flag,
        scope="conv1_batch_norm")
    conv1_relu = tf.nn.relu(conv1_bn, name='conv1_relu')

    pool1_in = tf.layers.max_pooling3d(
        inputs=conv1_relu, pool_size=2, strides=2, name='pool1')
    # pool1 (1, 48, 48, 48, 64)
    # pool1_frac = fractal_net(
    #     is_global_path_list[0],
    #     global_path_list[0],
    #     local_path_list[0],
    #     self.Blocks,
    #     self.Columns)(pool1_in)
    # pool1_old = pool1_in + pool1_frac
    pool1 = pool1_in
    conv2_1 = conv3d(
        input=pool1,
        output_chn=128,
        kernel_size=3,
        stride=1,
        use_bias=False,
        name='conv2')
    # (1, 48, 48, 48, 128)
    conv2_bn = tf.contrib.layers.batch_norm(
        conv2_1,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=phase_flag,
        scope="conv2_batch_norm")
    conv2_relu = tf.nn.relu(conv2_bn, name='conv2_relu')

    pool2_in = tf.layers.max_pooling3d(
        inputs=conv2_relu, pool_size=2, strides=2, name='pool2')
    # pool2  (1, 24, 24, 24, 128)
    # pool2_frac = fractal_net(
    #     is_global_path_list[1],
    #     global_path_list[1],
    #     local_path_list[1],
    #     self.Blocks,
    #     self.Columns)(pool2_in)
    # pool2 = pool2_in + pool2_frac
    pool2 = pool2_in

    conv3_1 = conv3d(
        input=pool2,
        output_chn=256,
        kernel_size=3,
        stride=1,
        use_bias=False,
        name='conv3a')
    # (1, 24, 24, 24, 256)
    conv3_1_bn = tf.contrib.layers.batch_norm(
        conv3_1,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=phase_flag,
        scope="conv3_1_batch_norm")
    conv3_1_relu = tf.nn.relu(conv3_1_bn, name='conv3_1_relu')
    conv3_2 = conv3d(
        input=conv3_1_relu,
        output_chn=256,
        kernel_size=3,
        stride=1,
        use_bias=False,
        name='conv3b')
    # (1, 24, 24, 24, 256)
    conv3_2 = conv3_2 + conv3_1
    conv3_2_bn = tf.contrib.layers.batch_norm(
        conv3_2,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=phase_flag,
        scope="conv3_2_batch_norm")
    conv3_2_relu = tf.nn.relu(conv3_2_bn, name='conv3_2_relu')

    pool3_in = tf.layers.max_pooling3d(
        inputs=conv3_2_relu, pool_size=2, strides=2, name='pool3')
    # pool3 (1, 12, 12, 12, 256)
    # pool3_frac = fractal_net(
    #     is_global_path_list[2],
    #     global_path_list[2],
    #     local_path_list[2],
    #     self.Blocks,
    #     self.Columns)(pool3_in)
    pool3 = pool3_in
    # pool3 = pool3_in + pool3_frac

    conv4_1 = conv3d(
        input=pool3,
        output_chn=512,
        kernel_size=3,
        stride=1,
        use_bias=False,
        name='conv4a')
    # conv4_1 (1, 12, 12, 12, 512)
    conv4_1_bn = tf.contrib.layers.batch_norm(
        conv4_1,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=phase_flag,
        scope="conv4_1_batch_norm")
    conv4_1_relu = tf.nn.relu(conv4_1_bn, name='conv4_1_relu')
    conv4_2 = conv3d(
        input=conv4_1_relu,
        output_chn=512,
        kernel_size=3,
        stride=1,
        use_bias=False,
        name='conv4b')
    conv4_2 = conv4_2 + conv4_1
    # conv4_2 (1, 12, 12, 12, 512)
    conv4_2_bn = tf.contrib.layers.batch_norm(
        conv4_2,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=phase_flag,
        scope="conv4_2_batch_norm")
    conv4_2_relu = tf.nn.relu(conv4_2_bn, name='conv4_2_relu')

    pool4 = tf.layers.max_pooling3d(
        inputs=conv4_2_relu,
        pool_size=2,
        strides=2,
        name='pool4')
    # pool4 (1, 6, 6, 6, 512)
    conv5_1 = conv_bn_relu(
        input=pool4,
        output_chn=512,
        kernel_size=3,
        stride=1,
        use_bias=False,
        is_training=phase_flag,
        name='conv5_1')
    # conv5_1 (1, 6, 6, 6, 512)
    conv5_2 = conv_bn_relu(
        input=conv5_1,
        output_chn=512,
        kernel_size=3,
        stride=1,
        use_bias=False,
        is_training=phase_flag,
        name='conv5_2')
    # conv5_2  (1, 6, 6, 6, 512)

    deconv1_1 = deconv_bn_relu(
        input=conv5_2,
        output_chn=512,
        is_training=phase_flag,
        name='deconv1_1')
    # (1, 12, 12, 12, 512)

    concat_1 = tf.concat([deconv1_1, conv4_2],
                         axis=concat_dim, name='concat_1')
    # (1, 12, 12, 12, 1024)

    deconv1_2_in = conv_bn_relu(
        input=concat_1,
        output_chn=256,
        kernel_size=3,
        stride=1,
        use_bias=False,
        is_training=phase_flag,
        name='deconv1_2')
    # deconv1_2_frac = fractal_net(
    #     is_global_path_list[3],
    #     global_path_list[3],
    #     local_path_list[3],
    #     self.Blocks,
    #     self.Columns)(deconv1_2_in)
    deconv1_2 = deconv1_2_in
    # deconv1_2 = deconv1_2_in + deconv1_2_frac  # (1, 12, 12, 12, 256)

    deconv2_1 = deconv_bn_relu(
        input=deconv1_2,
        output_chn=256,
        is_training=phase_flag,
        name='deconv2_1')

    concat_2 = tf.concat([deconv2_1, conv3_2],
                         axis=concat_dim, name='concat_2')
    # deconv2_2 (1, 24, 24, 24, 512)
    deconv2_2_in = conv_bn_relu(
        input=concat_2,
        output_chn=128,
        kernel_size=3,
        stride=1,
        use_bias=False,
        is_training=phase_flag,
        name='deconv2_2')
    # deconv2_2_frac = fractal_net(
    #     is_global_path_list[4],
    #     global_path_list[4],
    #     local_path_list[4],
    #     self.Blocks,
    #     self.Columns)(deconv2_2_in)
    deconv2_2 = deconv2_2_in
    # deconv2_2 = deconv2_2_in + deconv2_2_frac
    # deconv2_2 (1, 24, 24, 24, 128)

    deconv3_1 = deconv_bn_relu(
        input=deconv2_2,
        output_chn=128,
        is_training=phase_flag,
        name='deconv3_1')
    # deconv3_1 (1, 48, 48, 48, 128)

    concat_3 = tf.concat([deconv3_1, conv2_1],
                         axis=concat_dim, name='concat_3')
    # deconv3_1 (1, 48, 48, 48, 256)

    deconv3_2_in = conv_bn_relu(
        input=concat_3,
        output_chn=64,
        kernel_size=3,
        stride=1,
        use_bias=False,
        is_training=phase_flag,
        name='deconv3_2')
    # deconv3_2_frac = fractal_net(
    #     is_global_path_list[5],
    #     global_path_list[5],
    #     local_path_list[5],
    #     self.Blocks,
    #     self.Columns)(deconv3_2_in)
    deconv3_2 = deconv3_2_in
    # deconv3_2 = deconv3_2_in + deconv3_2_frac
    # deconv3_2 (1, 48, 48, 48, 64)

    deconv4_1 = deconv_bn_relu(
        input=deconv3_2,
        output_chn=64,
        is_training=phase_flag,
        name='deconv4_1')
    # deconv4_2 (1, 96, 96, 96, 32)

    concat_4 = tf.concat([deconv4_1, conv1_1],
                         axis=concat_dim, name='concat_4')
    # deconv4_2 (1, 96, 96, 96, 64)
    deconv4_2 = conv_bn_relu(
        input=concat_4,
        output_chn=32,
        kernel_size=3,
        stride=1,
        use_bias=False,
        is_training=phase_flag,
        name='deconv4_2')  # deconv4_2 (1, 96, 96, 96, 32)

    pre_pro = conv3d(
        input=deconv4_2,
        output_chn=output_channel,
        kernel_size=1,
        stride=1,
        use_bias=True,
        name='pre_pro')
    # pred_frac = fractal_net(is_global_path_list[3],global_path_list[3],local_path_list[3],self.Blocks,self.Columns)(pre_pro)
    pred_prob = pre_pro  # pred_prob (1, 96, 96, 96, 8)  Here get the prediction

    # ======================For predicition=============================
    # auxiliary prediction 0
    aux0_conv = conv3d(
        input=deconv1_2,
        output_chn=output_channel,
        kernel_size=1,
        stride=1,
        use_bias=True,
        name='aux0_conv')  # aux0_conv (1, 12, 12, 12, 8) 8 class output
    aux0_deconv_1 = Deconv3d(
        input=aux0_conv,
        output_chn=output_channel,
        name='aux0_deconv_1')  # aux0_deconv_1 (1, 24, 24, 24, 8)
    aux0_deconv_2 = Deconv3d(
        input=aux0_deconv_1,
        output_chn=output_channel,
        name='aux0_deconv_2')  # aux0_deconv_2 (1, 48, 48, 48, 8)
    aux0_prob = Deconv3d(
        input=aux0_deconv_2,
        output_chn=output_channel,
        name='aux0_prob')  # aux0_prob (1, 96, 96, 96, 8)

    # auxiliary prediction 1
    aux1_conv = conv3d(
        input=deconv2_2,
        output_chn=output_channel,
        kernel_size=1,
        stride=1,
        use_bias=True,
        name='aux1_conv')  # aux1_conv (1, 24, 24, 24, 8)
    aux1_deconv_1 = Deconv3d(
        input=aux1_conv,
        output_chn=output_channel,
        name='aux1_deconv_1')  # aux1_deconv_1 (1, 48, 48, 48, 8)
    aux1_prob = Deconv3d(
        input=aux1_deconv_1,
        output_chn=output_channel,
        name='aux1_prob')  # aux1_prob (1, 96, 96, 96, 8)

    # auxiliary prediction 2
    aux2_conv = conv3d(
        input=deconv3_2,
        output_chn=output_channel,
        kernel_size=1,
        stride=1,
        use_bias=True,
        name='aux2_conv')  # aux2_conv (1, 48, 48, 48, 8)
    aux2_prob = Deconv3d(
        input=aux2_conv,
        output_chn=output_channel,
        name='aux2_prob')  # aux2_prob (1, 96, 96, 96, 8)

    soft_prob = tf.nn.softmax(pred_prob, name='pred_soft')
    pred_label = tf.argmax(soft_prob, axis=4, name='argmax')

    return pred_prob, pred_label, aux0_prob, aux1_prob, aux2_prob

def unet_resnet(input_pred, input_img, output_channel, stage):
    input_shape = input_img.shape
    input_channel = input_shape.dims[-1].value
    input_pred_softmax = tf.nn.softmax(input_pred, name='softmax_ss' + stage)
    forground_input_pred = tf.expand_dims(input_pred_softmax[:, :, :, :, 1], axis=-1)
    input_concat = tf.concat([forground_input_pred, input_img], axis=-1)  # (1, 96, 96, 96, 2)

    input_attention = forground_input_pred * input_img  # (1, 96, 96, 96, input_channel)
    # conv block1
    conv_bn_1_1 = conv_bn_relu(input=input_attention, output_chn=16, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block1_conv1')
    input_cat = tf.concat([input_attention, input_attention, input_attention, input_attention,
                           input_attention, input_attention, input_attention, input_attention], axis=-1)
    # diffirence for odd input or even input
    if input_channel % 2 == 0 or input_channel == 1:
        input_tile = tf.tile(input=input_attention, multiples=[1, 1, 1, 1, int(16/input_channel)], name='tile' + stage)
    else:
        input_tile = tf.tile(input=input_attention, multiples=[1, 1, 1, 1, int(16/(input_channel-1))], name='tile' + stage)
        input_tile = input_tile[:,:,:,:,0:16]
    conv_bn_skip_1_1 = input_tile + conv_bn_1_1
    pool1_1 = tf.layers.max_pooling3d(inputs=conv_bn_skip_1_1, pool_size=2, strides=2, name=stage + 'pool_1_1')

    # conv block2
    conv_bn_2_1 = conv_bn_relu(input=pool1_1, output_chn=32, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block2_conv1')
    conv_bn_2_2 = conv_bn_relu(input=conv_bn_2_1, output_chn=32, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block2_conv2')
    pool1_1_cat = tf.concat([pool1_1, pool1_1], axis=-1)
    conv_bn_skip_2_1 = pool1_1_cat + conv_bn_2_2
    pool_2_1 = tf.layers.max_pooling3d(inputs=conv_bn_skip_2_1, pool_size=2, strides=2, name=stage + 'pool2_2')

    # conv block3
    conv_bn_3_1 = conv_bn_relu(input=pool_2_1, output_chn=64, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block3_conv1')
    conv_bn_3_2 = conv_bn_relu(input=conv_bn_3_1, output_chn=64, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block3_conv2')
    conv_bn_3_3 = conv_bn_relu(input=conv_bn_3_2, output_chn=64, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block3_conv3')
    pool_2_1_cat = tf.concat([pool_2_1, pool_2_1], axis=-1)
    conv_bn_skip_3_1 = conv_bn_3_3 + pool_2_1_cat
    pool3_1 = tf.layers.max_pooling3d(inputs=conv_bn_skip_3_1, pool_size=2, strides=2, name=stage + 'pool3_1')

    # conv block4
    conv_bn_4_1 = conv_bn_relu(input=pool3_1, output_chn=128, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block4_conv1')
    conv_bn_4_2 = conv_bn_relu(input=conv_bn_4_1, output_chn=128, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block4_conv2')
    conv_bn_4_3 = conv_bn_relu(input=conv_bn_4_2, output_chn=128, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block4_conv3')
    pool3_1_cat = tf.concat([pool3_1, pool3_1], axis=-1)
    conv_bn_skip_4_1 = conv_bn_4_3 + pool3_1_cat
    pool4_1 = tf.layers.max_pooling3d(inputs=conv_bn_skip_4_1, pool_size=2, strides=2, name=stage + 'pool4_1')

    # conv block5
    conv_bn_5_1 = conv_bn_relu(input=pool4_1, output_chn=256, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block5_conv1')
    conv_bn_5_2 = conv_bn_relu(input=conv_bn_5_1, output_chn=256, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block5_conv2')
    conv_bn_5_3 = conv_bn_relu(input=conv_bn_5_2, output_chn=256, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block5_conv3')
    pool4_1_cat = tf.concat([pool4_1, pool4_1], axis=-1)
    conv_bn_skip_5_1 = conv_bn_5_3 + pool4_1_cat

    # upsampling conv block6
    deconv_bn_1_1 = deconv_bn_relu(input=conv_bn_skip_5_1, output_chn=128, is_training=True,
                                   name=stage + 'deconv_1_1')
    concat1 = tf.concat([deconv_bn_1_1, conv_bn_skip_4_1], axis=-1, name=stage + 'concat1')
    conv_bn_6_1 = conv_bn_relu(input=concat1, output_chn=256, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block6_conv1')
    conv_bn_6_2 = conv_bn_relu(input=conv_bn_6_1, output_chn=256, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block6_conv2')
    conv_bn_6_3 = conv_bn_relu(input=conv_bn_6_2, output_chn=256, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'blovk6_conv3')

    deconv_bn_1_1_cat = tf.concat([deconv_bn_1_1, deconv_bn_1_1], axis=-1)
    conv_bn_skip_6_1 = conv_bn_6_3 + deconv_bn_1_1_cat

    # conv block7
    deconv_bn_2_1 = deconv_bn_relu(input=conv_bn_skip_6_1, output_chn=64, is_training=True,
                                   name=stage + 'deconv_2_1')
    concat2 = tf.concat([deconv_bn_2_1, conv_bn_skip_3_1], axis=-1, name=stage + 'concat2')
    conv_bn_7_1 = conv_bn_relu(input=concat2, output_chn=128, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block7_conv1')
    conv_bn_7_2 = conv_bn_relu(input=conv_bn_7_1, output_chn=128, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block7_conv2')
    conv_bn_7_3 = conv_bn_relu(input=conv_bn_7_2, output_chn=128, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block7_conv3')
    deconv_bn_2_1_cat = tf.concat([deconv_bn_2_1, deconv_bn_2_1], axis=-1)
    conv_bn_skip_7_1 = conv_bn_7_3 + deconv_bn_2_1_cat

    # conv block8
    deconv_bn_3_1 = deconv_bn_relu(input=conv_bn_skip_7_1, output_chn=32, is_training=True,
                                   name=stage + 'deconv_3_1')
    concat3 = tf.concat([deconv_bn_3_1, conv_bn_skip_2_1], axis=-1, name=stage + 'concat3')
    conv_bn_8_1 = conv_bn_relu(input=concat3, output_chn=64, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block8_conv1')
    conv_bn_8_2 = conv_bn_relu(input=conv_bn_8_1, output_chn=64, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block8_conv2')

    deconv_bn_3_1_cat = tf.concat([deconv_bn_3_1, deconv_bn_3_1], axis=-1)
    conv_bn_skip_8_1 = conv_bn_8_2 + deconv_bn_3_1_cat

    # conv block9
    deconv_bn_4_1 = deconv_bn_relu(input=conv_bn_skip_8_1, output_chn=16, is_training=True,
                                   name=stage + 'deconv_4_1')
    concat4 = tf.concat([deconv_bn_4_1, conv_bn_skip_1_1], axis=-1, name=stage + 'conca4_1')
    conv_bn_9_1 = conv_bn_relu(input=concat4, output_chn=32, kernel_size=3, stride=1, use_bias=False,
                               is_training=True, name=stage + 'block9_conv1')
    deconv_bn_4_1_cat = tf.concat([deconv_bn_4_1, deconv_bn_4_1], axis=-1)
    conv_bn_skip_9_1 = conv_bn_9_1 + deconv_bn_4_1_cat

    # prediction layer
    pred = conv3d(input=conv_bn_skip_9_1, output_chn=output_channel, kernel_size=1, stride=1, use_bias=True,
                  name=stage + 'pred')
    soft_prob_v = tf.nn.softmax(pred, name='pred_soft_v')
    pred_label_v = tf.argmax(soft_prob_v, axis=4, name='argmax_v')
    return pred, pred_label_v

def conv3d(
        input,
        output_chn,
        kernel_size,
        stride,
        use_bias=False,
        name='conv'):
    return tf.layers.conv3d(
        inputs=input,
        filters=output_chn,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        data_format='channels_last',
        kernel_initializer=tf.truncated_normal_initializer(
            0.0,
            0.01),
        kernel_regularizer=slim.l2_regularizer(0.0005),
        use_bias=use_bias,
        name=name)


def conv_bn_relu(
        input,
        output_chn,
        kernel_size,
        stride,
        use_bias,
        is_training,
        name):
    with tf.variable_scope(name):
        conv = conv3d(
            input,
            output_chn,
            kernel_size,
            stride,
            use_bias,
            name='conv')
        bn = tf.contrib.layers.batch_norm(
            conv,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            is_training=is_training,
            scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu


# deconvolution
def Deconv3d(input, output_chn, name):
    batch, in_depth, in_height, in_width, in_channels = [
        int(d) for d in input.get_shape()]
    filter = tf.get_variable(
        name + "/filter",
        shape=[
            4,
            4,
            4,
            output_chn,
            in_channels],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
            0,
            0.01),
        regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv3d_transpose(
        value=input,
        filter=filter,
        output_shape=[
            batch,
            in_depth * 2,
            in_height * 2,
            in_width * 2,
            output_chn],
        strides=[
            1,
            2,
            2,
            2,
            1],
        padding="SAME",
        name=name)
    return conv


def Unsample(input, output_chn, name):
    batch, in_depth, in_height, in_width, in_channels = [
        int(d) for d in input.get_shape()]
    base = input.shape[-2]
    data = 96 / int(base)
    print("base shape", data)
    filter = tf.get_variable(
        name + "/filter",
        shape=[
            4,
            4,
            4,
            output_chn,
            in_channels],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(
            0,
            0.01),
        regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv3d_transpose(
        value=input, filter=filter, output_shape=[
            batch, 96, 96, 96, output_chn], strides=[
            1, data, data, data, 1], padding="SAME", name=name)
    return conv


def deconv_bn_relu(input, output_chn, is_training, name):
    with tf.variable_scope(name):
        conv = Deconv3d(input, output_chn, name='deconv')
        # with tf.device("/cpu:0"):
        bn = tf.contrib.layers.batch_norm(
            conv,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            is_training=is_training,
            scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu


def conv_bn_relu_x3(
        input,
        output_chn,
        kernel_size,
        stride,
        use_bias,
        is_training,
        name):
    with tf.variable_scope(name):
        z = conv_bn_relu(
            input,
            output_chn,
            kernel_size,
            stride,
            use_bias,
            is_training,
            "dense1")
        z_out = conv_bn_relu(
            z,
            output_chn,
            kernel_size,
            stride,
            use_bias,
            is_training,
            "dense2")
        z_out = conv_bn_relu(
            z_out,
            output_chn,
            kernel_size,
            stride,
            use_bias,
            is_training,
            "dense3")
        return z + z_out