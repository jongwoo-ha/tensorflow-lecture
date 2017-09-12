import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, dropout, fully_connected
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

X_img = tf.reshape(X, [-1, 28, 28, 1])

L1 = dropout(max_pool2d(conv2d(X_img, 32, [3, 3]), [2, 2], padding='SAME'), keep, is_training=is_training)

L2 = dropout(max_pool2d(conv2d(L1, 64, [3, 3]), [2, 2], padding='SAME'), keep, is_training=is_training)

L3 = dropout(max_pool2d(conv2d(L2, 128, [3, 3]), [2, 2], padding='SAME'), keep, is_training=is_training)
L3 = tf.reshape(L3, [-1, 4*4*128])

L4 = dropout(fully_connected(L3, 512), keep, is_training=is_training)

logits = fully_connected(L4, 10, activation_fn=None)

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
            c, _ = sess.run([cost, train], {X: x_batch, Y: y_batch})
            cost_sum += c
    
        print('epoch: %2d, cost: %.5e' % ((epoch+1), cost_sum/num_batches))
    print('train:', datetime.datetime.now()-start)
    
    start = datetime.datetime.now()
    print('3-conv, 2-fc, contrib')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 1.64657e-01
# epoch:  2, cost: 4.40525e-02
# epoch:  3, cost: 3.21051e-02
# epoch:  4, cost: 2.17963e-02
# epoch:  5, cost: 1.80139e-02
# epoch:  6, cost: 1.67788e-02
# epoch:  7, cost: 1.23754e-02
# epoch:  8, cost: 1.16234e-02
# epoch:  9, cost: 9.16167e-03
# epoch: 10, cost: 1.03184e-02
# epoch: 11, cost: 6.07936e-03
# epoch: 12, cost: 7.74669e-03
# epoch: 13, cost: 6.89896e-03
# epoch: 14, cost: 4.93194e-03
# epoch: 15, cost: 6.03346e-03
# 3-conv, 2-fc, contrib
# accuracy: 0.99390
