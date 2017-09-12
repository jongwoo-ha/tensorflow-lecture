import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

learning_rate = 0.001
epochs = 15
batch_size = 100
keep = 0.75

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder_with_default(1.0, None)

init_w = tf.contrib.layers.xavier_initializer()
init_b = tf.zeros_initializer()

conv_W1 = tf.get_variable('conv_W1', [3, 3, 1, 32], initializer=init_w)
conv_W2 = tf.get_variable('conv_W2', [3, 3, 32, 64], initializer=init_w)
conv_W3 = tf.get_variable('conv_W3', [3, 3, 64, 128], initializer=init_w)

fc_W1 = tf.get_variable('fc_W1', [4*4*128, 512], initializer=init_w)
fc_W2 = tf.get_variable('fc_W2', [512, 10], initializer=init_w)

fc_b1 = tf.get_variable('fc_b1', [512], initializer=init_b)
fc_b2 = tf.get_variable('fc_b2', [10], initializer=init_b)



logits = tf.matmul(L4, fc_W2) + fc_b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 1)), tf.float32))

with tf.Session() as sess:
    start = datetime.datetime.now()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        cost_sum = 0
        num_batches = int(mnist.train.num_examples / batch_size)
    
        for i in range(num_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], {X: x_batch, Y: y_batch, keep_prob: keep})
            cost_sum += c
    
        print('epoch: %2d, cost: %.5e' % ((epoch+1), cost_sum/num_batches))
    print('train:', datetime.datetime.now()-start)

    start = datetime.datetime.now()
    print('3-conv, 2-fc')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 2.43190e-01
# epoch:  2, cost: 7.54674e-02
# epoch:  3, cost: 5.53051e-02
# epoch:  4, cost: 4.61846e-02
# epoch:  5, cost: 4.00658e-02
# epoch:  6, cost: 3.71918e-02
# epoch:  7, cost: 3.14704e-02
# epoch:  8, cost: 2.88506e-02
# epoch:  9, cost: 2.77171e-02
# epoch: 10, cost: 2.63208e-02
# epoch: 11, cost: 2.37383e-02
# epoch: 12, cost: 2.19633e-02
# epoch: 13, cost: 1.99412e-02
# epoch: 14, cost: 1.91553e-02
# epoch: 15, cost: 1.84868e-02
# 3-conv, 2-fc
# accuracy: 0.99310
