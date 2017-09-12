import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, dropout
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
is_training = tf.placeholder_with_default(False, [])

L1 = dropout(fully_connected(X, 512), keep, is_training=is_training)
L2 = dropout(fully_connected(L1, 512), keep, is_training=is_training)
L3 = dropout(fully_connected(L2, 512), keep, is_training=is_training)
L4 = dropout(fully_connected(L3, 512), keep, is_training=is_training)

logits = fully_connected(L4, 10, None)

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
            c, _ = sess.run([cost, train], {X: x_batch, Y: y_batch, is_training: True})
            cost_sum += c
    
        print('epoch: %2d, cost: %.5e' % ((epoch+1), cost_sum/num_batches))
    print('train:', datetime.datetime.now()-start)
    
    start = datetime.datetime.now()
    print('5 fully-connected, RELU, Xavier, dropout=%.2f, contrib' % keep)
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 3.00608e-01
# epoch:  2, cost: 1.34045e-01
# epoch:  3, cost: 1.05272e-01
# epoch:  4, cost: 8.68528e-02
# epoch:  5, cost: 7.49325e-02
# epoch:  6, cost: 7.02130e-02
# epoch:  7, cost: 5.89572e-02
# epoch:  8, cost: 5.59019e-02
# epoch:  9, cost: 4.81855e-02
# epoch: 10, cost: 4.64321e-02
# epoch: 11, cost: 4.46953e-02
# epoch: 12, cost: 4.27573e-02
# epoch: 13, cost: 3.87729e-02
# epoch: 14, cost: 3.94974e-02
# epoch: 15, cost: 3.70630e-02
# 5 fully-connected, RELU, Xavier, dropout=0.75, contrib
# accuracy: 0.98350
