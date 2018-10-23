import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time


mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

num_latent_variable = 100
num_hidden = 128
batch_size = 128
mnist_size = 784
num_label = 10
learning_rate = 0.0002
total_batch = int(mnist.train.num_examples / batch_size)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def plot(samples):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(5, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample+1)/2
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def generator(z, y,ngf=64):
    with tf.variable_scope("generator") as scope:
        inputs = tf.concat([z, y], axis=1)
        y_reshape = tf.reshape(y,[-1,1,1,num_label])


        inputs_dense = tf.layers.dense(inputs, units=1024, activation=tf.nn.relu)
        inputs = tf.concat([inputs_dense, y], axis=1)

        inputs_dense = tf.layers.dense(inputs, units=7*7*ngf*2,activation=tf.nn.relu)
        inputs_reshape = tf.reshape(inputs_dense,shape=[batch_size,7,7,ngf*2])

        conv1 = tf.concat([inputs_reshape , y_reshape*tf.ones([inputs_reshape.get_shape()[0], 7, 7 , num_label])], 3)
        conv1 = tcl.conv2d_transpose(inputs=conv1, num_outputs=ngf, activation_fn=tf.nn.relu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=tcl.batch_norm)

        conv2 = tf.concat([conv1 , y_reshape*tf.ones([conv1.get_shape()[0], 14, 14 , num_label])], 3)
        conv2 = tcl.conv2d_transpose(inputs=conv2, num_outputs=1, activation_fn=tf.nn.sigmoid, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02))

        return conv2


def discriminator(x, y,ndf=64,reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        y_reshape = tf.reshape(y,[-1,1,1,num_label])

        conv1 = tf.concat([x , y_reshape*tf.ones([x.get_shape()[0], 28, 28 , num_label])], 3)
        conv1 = tcl.conv2d(inputs=conv1, num_outputs=10, activation_fn=lrelu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02))

        conv2 = tf.concat([conv1 , y_reshape*tf.ones([conv1.get_shape()[0], 14, 14 , num_label])], 3)
        conv2 = tcl.conv2d(inputs=conv2, num_outputs=ndf, activation_fn=lrelu, stride=2, kernel_size=5, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_fn=tcl.batch_norm)

        Flatten_output = tf.layers.flatten(conv2)
        Flatten_output = tf.concat([Flatten_output,y],axis=1)

        Dense_output = tf.layers.dense(Flatten_output,units=1024)
        Dense_output = tf.concat([Dense_output,y],axis=1)

        output = tf.layers.dense(Dense_output,units=1)
        logits = tf.nn.sigmoid(output)

        return logits,output

def vars(name):
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

def init():
    X = tf.placeholder(tf.float32, shape=[batch_size, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])
    X_reshape = tf.reshape(X, shape = [-1,28,28,1])

    z_sample = tf.placeholder(tf.float32, shape=[None, 100])

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    G_output = generator(z_sample, Y)

    D_real,D_real_logits = discriminator(X_reshape, Y)
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

    label = tf.one_hot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10)
    label_8 = tf.one_hot([0, 1, 2, 3, 4, 5, 6, 7], 10)
    label = tf.concat([label, label, label, label, label,label,label, label,label,label, label,label,label_8], 0)

    label = sess.run(label)

    print("Start training batch_size : {} total_batch : {}".format(batch_size, total_batch))

    for epoch in range(10000):
        start =  time.time()
        for i in range(total_batch):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            z_variable = np.random.uniform(-1., 1., [batch_size, num_latent_variable])
            sess.run(d_optimizer, feed_dict={X: batch_X, Y: batch_Y, z_sample: z_variable})
            sess.run(g_optimizer, feed_dict={Y: batch_Y, z_sample: z_variable})



            if i % 100 == 0:
                d_l, g_l = sess.run([d_loss, g_loss],
                                    feed_dict={X: batch_X, Y: batch_Y,
                                               z_sample: np.random.uniform(-1., 1., [batch_size, num_latent_variable])})

                print("epoch : {} d_loss : {} g_loss : {} time : {}".format(epoch, d_l, g_l, time.time() - start))

                samples = sess.run(G_output,
                                       feed_dict={Y: label, z_sample: np.random.uniform(-1., 1., [batch_size, num_latent_variable])})
                fig = plot(samples[:50])
                plt.savefig('output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
                num_img += 10
                plt.close(fig)


def main():
    init()
    return


if __name__ == "__main__":
    main()
