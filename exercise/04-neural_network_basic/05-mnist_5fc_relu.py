import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

learning_rate = 0.001
epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

init_w = tf.random_normal_initializer()
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
            c, _ = sess.run([cost, train], {X: x_batch, Y: y_batch})
            cost_sum += c
    
        print('epoch: %2d, cost: %.5e' % ((epoch+1), cost_sum/num_batches))
    print('train:', datetime.datetime.now()-start)
    
    start = datetime.datetime.now()
    print('5 fully-connected, RELU')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 4.08253e+04
# epoch:  2, cost: 9.83264e+03
# epoch:  3, cost: 5.11212e+03
# epoch:  4, cost: 2.85237e+03
# epoch:  5, cost: 1.75679e+03
# epoch:  6, cost: 1.18792e+03
# epoch:  7, cost: 8.76172e+02
# epoch:  8, cost: 7.77383e+02
# epoch:  9, cost: 6.67766e+02
# epoch: 10, cost: 6.04672e+02
# epoch: 11, cost: 5.66789e+02
# epoch: 12, cost: 5.84894e+02
# epoch: 13, cost: 5.07389e+02
# epoch: 14, cost: 5.55090e+02
# epoch: 15, cost: 3.67704e+02
# 5 fully-connected, RELU
# accuracy: 0.95750
