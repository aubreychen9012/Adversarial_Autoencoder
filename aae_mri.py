import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import datetime
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data
from numpy import linalg as LA

# Parameters
input_dim = 28
n_l1 = 200
n_l2 = 400
z_dim = 35
batch_size = 100
n_epochs = 1000
learning_rate = 2e-5
beta1 = 0.9
retrain = 0
results_path = './Results/Adversarial_Autoencoder'

mnist = input_data.read_data_sets('./Data', one_hot=True)

def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    results_path = './Results/Adversarial_Autoencoder'
    folder_name = "/{0}_{1}_Adversarial_Autoencoder_Res". \
        format(datetime.datetime.now(), z_dim)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


# def generate_image_grid(sess, op):
#    """
#    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
#    :param sess: Tensorflow Session required to get the decoder output
#    :param op: Operation that needs to be called inorder to get the decoder output
#    :return: None, displays a matplotlib window with all the merged images.
#    """
#    x_points = np.arange(-10, 10, 1.5).astype(np.float32)
#    y_points = np.arange(-10, 10, 1.5).astype(np.float32)#

#    nx, ny = len(x_points), len(y_points)
#    plt.subplot()
#    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

#    for i, g in enumerate(gs):
#        z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
#        z = np.reshape(z, (1, 2))
#        x = sess.run(op, feed_dict={decoder_input: z})
#        ax = plt.subplot(g)
#        img = np.array(x.tolist()).reshape(28, 28)
#        ax.imshow(img, cmap='gray')
#        ax.set_xticks([])
#        ax.set_yticks([])
#        ax.set_aspect('auto')
#    plt.show()

def resblock(inputs, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='e_inputs')
        conv1 = Conv2d(input_layer, filters,(3,3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="e_conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='e_bn1')
        conv2 = Conv2d(conv1, filters, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="e_conv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                           gamma_init=gamma_init, name='e_bn2')
        conv_out = conv2.outputs+inputs
        conv_out = tf.nn.relu(conv_out)
    return conv_out

def ResBlock(inputs, filter_in, filter_out, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = Conv2d(input_layer, filter_in, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = Conv2d(conv1, filter_out, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(input_layer, filter_out, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')


        conv_out = conv2.outputs + conv3.outputs
    return conv_out


# image size /2
def ResBlockDown(inputs, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = Conv2d(input_layer, filters, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = Conv2d(conv1, filters*2, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(input_layer, filters*2, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')


        conv_out = conv2.outputs + conv3.outputs
    return conv_out


# image size *2
def ResBlockUp(inputs, input_size, batch_size, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = DeConv2d(input_layer, filters, (3, 3), (input_size*2,input_size*2), (2,2),
                         batch_size=batch_size,act=None, padding='SAME',
                         W_init=w_init, b_init=b_init, name="deconv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = DeConv2d(conv1, filters/2, (3, 3), (input_size*2,input_size*2), (1,1), act=None, padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="deconv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = DeConv2d(input_layer, filters/2, (3, 3), (input_size*2,input_size*2), (2,2), act=None, padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out

#
# def dense(x, n1, n2, name):
#     """
#     Used to create a dense layer.
#     :param x: input tensor to the dense layer
#     :param n1: no. of input neurons
#     :param n2: no. of output neurons
#     :param name: name of the entire dense layer.i.e, variable scope name.
#     :return: tensor with shape [batch_size, n2]
#     """
#     with tf.variable_scope(name, reuse=None):
#         weights = tf.get_variable("weights", shape=[n1, n2],
#                                   initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
#         bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
#         out = tf.add(tf.matmul(x, weights), bias, name='matmul')
#         return out

# The autoencoder network
def encoder(x, reuse=False, is_train=True):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    image_size = input_dim
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
    gf_dim = 16  # Dimension of gen filters in first conv layer. [64]
    ft_size = 3
    # c_dim = FLAGS.c_dim  # n_color 3
    # batch_size = 64  # 64
    with tf.variable_scope("Encoder", reuse=reuse):
        # x,y,z,_ = tf.shape(input_images)
        tl.layers.set_name_reuse(reuse)

        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.01)

        inputs = InputLayer(x, name='e_inputs')
        conv1 = Conv2d(inputs, gf_dim, (ft_size, ft_size), act=lambda x: tl.act.lrelu(x, 0.2), padding='SAME', W_init=w_init, b_init=b_init,
                       name="e_conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                               gamma_init=gamma_init, name='e_bn1')
        # image_size * image_size
        res1 = ResBlockDown(conv1.outputs, gf_dim, "res1", reuse, is_train)
        #res1 = tf.layers.conv2d(inputs=res1, filters = gf_dim*2, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res1_downsample')

        # s2*s2
        res2 = ResBlockDown(res1, gf_dim*2, "res2", reuse, is_train)
        #res2 = tf.layers.conv2d(inputs=res2, filters = gf_dim*4, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res2_downsample')

        # s4*s4
        res3 = ResBlockDown(res2, gf_dim*4, "res3", reuse, is_train)
        #res3 = tf.layers.conv2d(inputs=res3, filters = gf_dim*8, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res3_downsample')

        # s8*s8
        res4 = ResBlockDown(res3, gf_dim * 8, "res4", reuse, is_train)
        #res4 = tf.layers.conv2d(inputs=res4, filters = gf_dim*16, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res4_downsample')

        # s16*s16
        h_flat = tf.reshape(res4, shape=[-1, s16 * s16 * gf_dim*16])
        h_flat = InputLayer(h_flat, name='e_reshape')
        net_h = DenseLayer(h_flat, n_units=z_dim, act=tf.identity, name="e_dense_mean")
    return net_h.outputs

def decoder(x, reuse=False, is_train=True):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    image_size = input_dim
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
    gf_dim = 16  # Dimension of gen filters in first conv layer. [64]
    c_dim = 1  # n_color 3
    ft_size=3
    batch_size = 64  # 64
    with tf.variable_scope("Decoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        # gamma_init = tf.random_normal_initializer(1., 0.02)
        # weights_gener = dict()
        inputs = InputLayer(x, name='g_inputs')

        # s16*s16
        z_develop = DenseLayer(inputs, s16 * s16 * gf_dim*16, act=lambda x: tl.act.lrelu(x, 0.2), name='g_dense_z')
        z_develop = tf.reshape(z_develop.outputs, [-1,  s16, s16, gf_dim*16])
        z_develop = InputLayer(z_develop, name='g_reshape')
        conv1 = Conv2d(z_develop, gf_dim*8, (ft_size, ft_size),act=lambda x: tl.act.lrelu(x, 0.2) , padding='SAME',
                       W_init=w_init, b_init=b_init, name="g_conv1")

        # s16*s16
        res1 = ResBlockUp(conv1.outputs, s16, batch_size, gf_dim*8, "gres1", reuse, is_train)
        #res1 = tf.layers.conv2d_transpose(inputs=res1, filters = gf_dim*4, kernel_size = (ft_size, ft_size), strides = (2,2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                 kernel_initializer=w_init, trainable=True, name='res1_upsample')

        # s8*s8
        res2 = ResBlockUp(res1, s8, batch_size, gf_dim*4, "gres2", reuse, is_train)
        #res2 = tf.layers.conv2d_transpose(inputs=res2, filters=gf_dim*2, kernel_size=(ft_size, ft_size), strides=(2, 2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                  kernel_initializer=w_init, trainable=True, name='res2_upsample')

        # s4*s4
        res3 = ResBlockUp(res2,s4, batch_size, gf_dim*2, "gres3", reuse, is_train)
        #res3 = tf.layers.conv2d_transpose(inputs=res3, filters=gf_dim, kernel_size=(ft_size, ft_size), strides=(2, 2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                  kernel_initializer=w_init, trainable=True, name='res3_upsample')

        # s2*s2
        res4 = ResBlockUp(res3, s2, batch_size, gf_dim, "gres4", reuse, is_train)
        #res4 = tf.layers.conv2d_transpose(inputs=res4, filters=8, kernel_size=(ft_size, ft_size), strides=(2, 2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                  kernel_initializer=w_init, trainable=True, name='res4_upsample')
        # image_size*image_size
        res_inputs = InputLayer(res4, name='res_inputs')
        conv2 = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="g_conv2")
        conv2_std = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="g_conv2_std")

        # deconv1 = DeConv2d(res_inputs, c_dim, (3, 3), out_size=(image_size, image_size), strides=(1, 1),
        #                   padding="SAME", act=None, batch_size=batch_size, W_init=w_init, b_init=b_init,
        #                   name="g_mu_output")
        # deconv1 = DeConv2d(res_inputs, c_dim, (3, 3), out_size=(image_size, image_size), strides=(1, 1),
        #                   padding="SAME", act=None, batch_size=batch_size, W_init=w_init, b_init=b_init,
        #                   name="g_std_output")
        logits = conv1.outputs
    return conv2.outputs, conv2_std.outputs


def discriminator(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    w_init = tf.random_normal_initializer(stddev=0.01)
    # tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(x, name='dc/in')
        net_h0 = DenseLayer(net_in, n_units=n_l1,
                            W_init=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),
                            act=lambda x: tl.act.lrelu(x, 0.2), name='dc/h0/lin')
        net_h1 = DenseLayer(net_h0, n_units=n_l2,
                            W_init=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),
                            act=lambda x: tl.act.lrelu(x, 0.2), name='dc/h1/lin')
        net_h2 = DenseLayer(net_h1, n_units=1, W_init=w_init,
                            act=tf.identity, name='dc/h2/lin')
        logits = net_h2.outputs
        net_h2.outputs = tf.nn.sigmoid(net_h2.outputs)
        return net_h2.outputs, logits


def train(train_model=True,load=False, comment=None, model_name=None, modelstep=0):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """
    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim,1], name='Input')
        x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim,1], name='Target')
        
        real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
        decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')
        encoder_output = encoder(x_input, reuse=False, is_train=True)
        d_fake, d_fake_logits = discriminator(encoder_output, reuse=False)
        d_real, d_real_logits = discriminator(real_distribution, reuse=True)
        decoder_output = decoder(encoder_output, reuse=False, is_train=True)
        
        # test
        encoder_output_test = encoder(x_input, reuse=True, is_train=False)
        d_fake_test, d_fake_logits_test = discriminator(encoder_output_test, reuse=True)
        d_real_test, d_real_logits_test = discriminator(real_distribution, reuse=True)
        decoder_output_test = decoder(encoder_output, reuse=True, is_train=False)

        decoder_image = decoder(decoder_input, reuse=True, is_train=False)

        # Autoencoder loss
        autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))
        autoencoder_loss_test = tf.reduce_mean(tf.square(x_target - decoder_output_test))

        # Discrimminator Loss
        tf_randn_real = tf.random_uniform(tf.shape(d_real),0.8,1.1)
        tf_randn_fake = tf.random_uniform(tf.shape(d_real), 0.0, 0.1)
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real, logits=d_real_logits))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_fake, logits=d_fake_logits))
        dc_loss = dc_loss_fake + dc_loss_real
        
        dc_loss_real_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real, logits=d_real_logits_test))
        dc_loss_fake_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake_logits_test))
        dc_loss_test = dc_loss_fake_test + dc_loss_real_test
    
        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake_logits))
        generator_loss_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_test), logits=d_fake_logits_test))

        all_variables = tf.trainable_variables()
        dc_var = tl.layers.get_variables_with_name('Discriminator', True, True)
        en_var = tl.layers.get_variables_with_name('Encoder', True, True)

        var_grad_autoencoder = tf.gradients(autoencoder_loss, all_variables)[0]
        var_grad_discriminator = tf.gradients(dc_loss, dc_var)[0]
        var_grad_generator = tf.gradients(generator_loss, en_var)[0]

        # Optimizers
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                         beta1=beta1).minimize(dc_loss, var_list=dc_var)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                     beta1=beta1).minimize(generator_loss, var_list=en_var)
        autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                       beta1=beta1).minimize(autoencoder_loss)

        tl.layers.initialize_global_variables(sess)
        # Reshape immages to display them
        input_images = tf.reshape(x_input, [-1, input_dim, input_dim, 1])
        generated_images = tf.reshape(decoder_output, [-1, input_dim, input_dim, 1])
        # generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])
        tensorboard_path, saved_model_path, log_path = form_results()
        writer = tf.summary.FileWriter(logdir= tensorboard_path, graph=sess.graph)
        # Tensorboard visualization
        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        #tf.summary.scalar(name='Autoencoder Test Loss', tensor=autoencoder_loss_test)
        tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
        #tf.summary.scalar(name='Discriminator Test Loss', tensor=dc_loss_test)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        #tf.summary.scalar(name='Generator Test Loss', tensor=generator_loss_test)
        
        tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
        tf.summary.histogram(name='Encoder Test Distribution', values=encoder_output_test)
        
        tf.summary.histogram(name='Real Distribution', values=real_distribution)
        tf.summary.histogram(name='Gradient AE', values=var_grad_autoencoder)
        tf.summary.histogram(name='Gradient D', values=var_grad_discriminator)
        tf.summary.histogram(name='Gradient G', values=var_grad_generator)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
        tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
    # Saving the model

    step = 0
    #with tf.Session() as sess:
    if train_model:
        with open(log_path + '/log.txt', 'a') as log:
            log.write("Comment: {}\n".format(comment))
            log.write("\n")
            log.write("input_dim: {}\n".format(input_dim))
            log.write("n_l1: {}\n".format(n_l1))
            log.write("n_l2: {}\n".format(n_l2))
            log.write("z_dim: {}\n".format(z_dim))
            log.write("batch_size: {}\n".format(batch_size))
            log.write("learning_rate: {}\n".format(learning_rate))
            log.write("beta1: {}\n".format(beta1))
            log.write("\n")
        if load:
            saver = tf.train.import_meta_graph(
                "./Results/Adversarial_Autoencoder/"+str(model_name)+"/Saved_models/" + str(modelstep) + ".meta")
            saver.restore(sess,
                          "./Results/Adversarial_Autoencoder/"+str(model_name)+"/Saved_models/"
                          + str(modelstep))
            # saver.restore(sess,results_path + '/' + str(model_name) + '/Saved_models/'+str(modelstep)+".meta")

        for i in range(n_epochs):
            n_batches = int(mnist.train.num_examples / batch_size)
            #b = 0
            for b in range(1, n_batches + 1):
                batch_x, _ = mnist.train.next_batch(batch_size)
                batch_x = batch_x.reshape(batch_size, 28,28,1)

                z_real_dist = np.random.normal(0, 1, (batch_size, z_dim)) * 1.
                z_real_dist = z_real_dist.astype("float32")
                sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                #for r in range(2):
                sess.run(discriminator_optimizer,
                             feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})
                sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                if b % 50 == 0:
                    a_loss, d_loss, g_loss, a_grad, d_grad, g_grad, en_output, dreal, dfake, de_output, summary \
                        = sess.run(
                        [autoencoder_loss, dc_loss, generator_loss,
                         var_grad_autoencoder, var_grad_discriminator, var_grad_generator,
                         encoder_output, d_real, d_fake, decoder_output, summary_op],
                        feed_dict={x_input: batch_x, x_target: batch_x,
                                   real_distribution: z_real_dist})
                    print("ae gradient norm:{}, d gradient norm:{}, g gradient norm:{}".format(LA.norm(a_grad), LA.norm(d_grad), LA.norm(g_grad)))
                    writer.add_summary(summary, global_step=step)
                    print("Epoch: {}, iteration: {}".format(i, b))
                    print("Autoencoder Loss: {}".format(a_loss))
                    print("Discriminator Loss: {}".format(d_loss))
                    print("Generator Loss: {}".format(g_loss))
                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(i, b))
                        log.write("Autoencoder Loss: {}\n".format(a_loss))
                        log.write("Discriminator Loss: {}\n".format(d_loss))
                        log.write("Generator Loss: {}\n".format(g_loss))
                step += 1
            saver.save(sess, save_path=saved_model_path, global_step=step)

    else:
        # Get the latest results folder
        all_results = os.listdir(results_path)
        all_results.sort()
        saver.restore(sess,
                      save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=int, default=0, help='retrain model')
    parser.add_argument('--model_name', type=str, default='None', help='model to retrain on')
    parser.add_argument('--step', type=str, default='None', help='model to retrain on')
    parser.add_argument('--comment', type=str, default='None', help='model comment')
    args = parser.parse_args()
    train(train_model=True,load =args.load, comment=args.comment, model_name=args.model_name, modelstep=args.step)
