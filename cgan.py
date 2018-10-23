import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

num_latent_variable = 100
num_hidden = 128
batch_size = 64
mnist_size = 784
num_label = 10
learning_rate = 0.0001
total_batch = int(mnist.train.num_examples / batch_size)


def plot(samples):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(5, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28,28),cmap='Greys_r')
    return fig

d_weight_1 = tf.Variable(tf.random_normal(shape=[mnist_size+num_label, num_hidden], stddev=5e-2))
d_bias_1 = tf.Variable(tf.constant(0.0,shape=[num_hidden]))
    
d_weight_2 = tf.Variable(tf.random_normal(shape=[num_hidden, 1], stddev=5e-2))
d_bias_2 = tf.Variable(tf.constant(0.0,shape=[1]))


g_weight_1 = tf.Variable(tf.random_normal(shape=[num_latent_variable+num_label, num_hidden], stddev=5e-2))
g_bias_1 = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
    
g_weight_2 = tf.Variable(tf.random_normal(shape=[num_hidden, mnist_size], stddev=5e-2))
g_bias_2 = tf.Variable(tf.constant(0.0, shape=[mnist_size]))


def generator(z,y):
    
    inputs = tf.concat([z,y] , axis = 1)
    
    hidden_layer = tf.nn.relu((tf.matmul(inputs, g_weight_1) + g_bias_1))
    output_layer = tf.matmul(hidden_layer, g_weight_2) + g_bias_2
    output = tf.nn.sigmoid(output_layer)
    
    return output

def discriminator(x,y):
    
    inputs = tf.concat([x,y] , axis = 1)

    hidden_layer = tf.nn.relu((tf.matmul(inputs, d_weight_1) + d_bias_1))
    logits = tf.matmul(hidden_layer, d_weight_2) + d_bias_2
    predicted_value = tf.nn.sigmoid(logits)
    
    return predicted_value, logits



def init():
    X = tf.placeholder(tf.float32, shape=[None,784])
    Y = tf.placeholder(tf.float32, shape=[None,10])
    
    z_sample = tf.placeholder(tf.float32, shape=[None,100])
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    G_output = generator(z_sample,Y)
    
    
    D_real, D_logit_real = discriminator(X,Y)
    D_fake, D_logit_fake = discriminator(G_output,Y)
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    
    d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=[d_weight_1,d_bias_1,d_weight_2,d_bias_2])
    g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=[g_weight_1,g_bias_1,g_weight_2,g_bias_2])
    
    sess.run(tf.global_variables_initializer())
    
    num_img = 0
    
    label = tf.one_hot([0, 1, 2,3,4,5,6,7,8,9], 10)
    label = tf.concat([label,label,label,label,label],0)
    label = sess.run(label)
    
    print("Start training batch_size : {} total_batch : {}".format(batch_size,total_batch))
    
    for epoch in range(100):
        
        for i in range(total_batch):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            z_variable = np.random.uniform(-1., 1., [batch_size, 100])
            sess.run(d_optimizer,feed_dict={ X : batch_X, Y:batch_Y, z_sample : z_variable })
            sess.run(g_optimizer,feed_dict={ Y:batch_Y, z_sample : z_variable })
        
        d_l,g_l = sess.run([d_loss,g_loss],feed_dict={X : batch_X, Y:batch_Y, z_sample : np.random.uniform(-1., 1., [batch_size, 100])})
        print("epoch : {} d_loss : {} g_loss : {}".format(epoch,d_l,g_l))
        
        if epoch %10 ==0:
            samples = sess.run(G_output, feed_dict={Y:label,z_sample: np.random.uniform(-1., 1., [50, num_latent_variable])})
            fig = plot(samples)
            plt.savefig('output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 10
            plt.close(fig)

def main():
    init()
    return

if __name__=="__main__":
    main()
    