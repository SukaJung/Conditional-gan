import tensorflow as tf
import tensorflow.contrib.layers as tcl

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
from dataset import *
num_latent_variable = 100
num_hidden = 128
batch_size = 128
num_label = 2
learning_rate = 0.0002

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def plot(label1,label2):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(5, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(label1):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample*255)
        sample = sample.astype(np.uint8)
        plt.imshow(sample.reshape(64, 64 ,3))
    for i, sample in enumerate(label2):
        ax = plt.subplot(gs[25+i])
        plt.axis('off')
        sample = (sample*255)
        sample = sample.astype(np.uint8)
        plt.imshow(sample.reshape(64, 64 ,3))
    return fig


def generator(z, y,ngf=64):
    with tf.variable_scope("generator") as scope:
        inputs = tf.concat([z, y], axis=1)
        y_reshape = tf.reshape(y,[-1,1,1,num_label])

        inputs_dense = tf.layers.dense(inputs, units=1024, activation=tf.nn.relu)
        inputs = tf.concat([inputs_dense, y], axis=1)

        inputs_dense = tf.layers.dense(inputs, units=4*4*ngf*2,activation=tf.nn.relu)
        inputs_reshape = tf.reshape(inputs_dense,shape=[batch_size,4,4,ngf*2])

        conv1 = tf.concat([inputs_reshape , y_reshape*tf.ones([inputs_reshape.get_shape()[0], 4, 4 , num_label])], 3)
        conv1 = tcl.conv2d_transpose(inputs=conv1, num_outputs=ngf, activation_fn=tf.nn.relu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=tcl.batch_norm)

        conv2 = tf.concat([conv1 , y_reshape*tf.ones([conv1.get_shape()[0], 8, 8 , num_label])], 3)
        conv2 = tcl.conv2d_transpose(inputs=conv2, num_outputs=ngf*4, activation_fn=tf.nn.relu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02))
        
        conv3 = tf.concat([conv2 , y_reshape*tf.ones([conv2.get_shape()[0], 16, 16 , num_label])], 3)
        conv3 = tcl.conv2d_transpose(inputs=conv3, num_outputs=ngf*8, activation_fn=tf.nn.relu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02))
        
        conv4 = tf.concat([conv3 , y_reshape*tf.ones([conv3.get_shape()[0], 32, 32 , num_label])], 3)
        conv4 = tcl.conv2d_transpose(inputs=conv4, num_outputs=3, activation_fn=tf.nn.sigmoid, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02))

        return conv4


def discriminator(x, y,ndf=64,reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        y_reshape = tf.reshape(y,[-1,1,1,num_label])
        print(x.shape)
        conv1 = tf.concat([x , y_reshape*tf.ones([x.get_shape()[0], 64, 64 , num_label])], 3)
        conv1 = tcl.conv2d(inputs=conv1, num_outputs=ndf*8, activation_fn=lrelu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02))

        conv2 = tf.concat([conv1 , y_reshape*tf.ones([conv1.get_shape()[0], 32, 32 , num_label])], 3)
        conv2 = tcl.conv2d(inputs=conv2, num_outputs=ndf*4, activation_fn=lrelu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=tcl.batch_norm)
        
        conv3 = tf.concat([conv2 , y_reshape*tf.ones([conv2.get_shape()[0], 16, 16 , num_label])], 3)
        conv3 = tcl.conv2d(inputs=conv3, num_outputs=ndf*2, activation_fn=lrelu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=tcl.batch_norm)
        
        conv4 = tf.concat([conv3 , y_reshape*tf.ones([conv3.get_shape()[0], 8, 8 , num_label])], 3)
        conv4 = tcl.conv2d(inputs=conv4, num_outputs=ndf, activation_fn=lrelu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=tcl.batch_norm)

        Flatten_output = tf.layers.flatten(conv4)
        Flatten_output = tf.concat([Flatten_output,y],axis=1)

        Dense_output = tf.layers.dense(Flatten_output,units=1024)
        Dense_output = tf.concat([Dense_output,y],axis=1)

        output = tf.layers.dense(Dense_output,units=1)
        logits = tf.nn.sigmoid(output)

        return logits,output

def vars(name):
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

def init(data):
    X = tf.placeholder(tf.float32, shape=[batch_size, 64,64,3])
    Y = tf.placeholder(tf.float32, shape=[None, 2])

    z_sample = tf.placeholder(tf.float32, shape=[None, 100])

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    G_output = generator(z_sample, Y)

    D_real,D_real_logits = discriminator(X, Y)
    D_fake,D_fake_logits = discriminator(G_output, Y,reuse=True)

    d_fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_fake), logits=D_fake_logits))
    d_real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_real), logits=D_real_logits))

    d_loss = d_fake_loss+d_real_loss
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_fake), logits=D_fake_logits))

    t_vars = tf.trainable_variables()
    d_var = [var for var in t_vars if 'dis' in var.name]
    g_var = [var for var in t_vars if 'gen' in var.name]

    d_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(d_loss, var_list=d_var)
    g_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(g_loss, var_list=g_var)

    sess.run(tf.global_variables_initializer())

    num_img = 0

    label = tf.one_hot([0, 1], 2)
    label = np.zeros([128,2])
    label[:64,0] = 1
    label[64:,1] = 1
    
    total_batch = int(data.cn/batch_size)

    print("Start training batch_size : {} total_batch : {}".format(batch_size, total_batch))
    test_latent_variable = np.random.uniform(-1., 1., [batch_size, num_latent_variable])
    for epoch in range(10000):
        start =  time.time()
        for i in range(total_batch):
            batch_X, batch_Y = data.getbatch(batch_size)
            z_variable = np.random.uniform(-1., 1., [batch_size, num_latent_variable])
            sess.run(d_optimizer, feed_dict={X: batch_X, Y: batch_Y, z_sample: z_variable})
            sess.run(g_optimizer, feed_dict={Y: batch_Y, z_sample: z_variable})



#         if epoch % 5 == 0:
        d_l, g_l = sess.run([d_loss, g_loss], feed_dict={X: batch_X,Y: batch_Y,z_sample: np.random.uniform(-1., 1., [batch_size, num_latent_variable])})

        print("epoch : {} d_loss : {} g_loss : {} time : {}".format(epoch, d_l, g_l, time.time() - start))

        samples = sess.run(G_output,
                                      feed_dict={Y: label, z_sample: test_latent_variable})
        fig = plot(samples[:25],samples[64:89])
        plt.savefig('output/%s.png' % str(epoch).zfill(4), bbox_inches='tight')
#             num_img += 5
        plt.close(fig)


def main():
    data = DCDataset(data_dir="/home/suka/dataset/preprocessed_catdog/",index_dir="/home/suka/PycharmProjects/cgan/")
    init(data)
    return


if __name__ == "__main__":
    main()
