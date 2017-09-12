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
keep_prob = tf.placeholder_with_default(1., [])

init_w = tf.contrib.layers.xavier_initializer()
init_b = tf.zeros_initializer()

W1 = tf.get_variable('W1', [784, 512], initializer=init_w)
W2 = tf.get_variable('W2', [512, 512], initializer=init_w)
W3 = tf.get_variable('W3', [512, 512], initializer=init_w)
W4 = tf.get_variable('W4', [512, 512], initializer=init_w)
W5 = tf.get_variable('W5', [512, 10], initializer=init_w)

b1 = tf.get_variable('b1', [512], initializer=init_b)
b2 = tf.get_variable('b2', [512], initializer=init_b)
b3 = tf.get_variable('b3', [512], initializer=init_b)
b4 = tf.get_variable('b4', [512], initializer=init_b)
b5 = tf.get_variable('b5', [10], initializer=init_b)

L1 = None
L2 = None
L3 = None
L4 = None

logits = tf.matmul(L4, W5) + b5

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
    print('5 fully-connected, RELU, Xavier, dropout=%.2f' % keep)
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 2.98607e-01
# epoch:  2, cost: 1.35218e-01
# epoch:  3, cost: 1.04106e-01
# epoch:  4, cost: 8.77811e-02
# epoch:  5, cost: 7.55074e-02
# epoch:  6, cost: 6.43376e-02
# epoch:  7, cost: 6.09853e-02
# epoch:  8, cost: 5.40211e-02
# epoch:  9, cost: 5.20516e-02
# epoch: 10, cost: 4.83009e-02
# epoch: 11, cost: 4.25152e-02
# epoch: 12, cost: 4.03369e-02
# epoch: 13, cost: 4.29661e-02
# epoch: 14, cost: 4.02640e-02
# epoch: 15, cost: 3.72321e-02
# 5 fully-connected, RELU, Xavier, dropout=0.75
# accuracy: 0.98270
