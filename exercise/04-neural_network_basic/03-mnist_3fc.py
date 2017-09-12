import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

learning_rate = 0.001
epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [])
Y = tf.placeholder(tf.float32, [])

init_w = tf.random_normal_initializer()
init_b = tf.zeros_initializer()

W1 = tf.get_variable('W1', [], initializer=init_w)
W2 = tf.get_variable('W2', [512, 512], initializer=init_w)
W3 = tf.get_variable('W3', [], initializer=init_w)

b1 = tf.get_variable('b1', [], initializer=init_b)
b2 = tf.get_variable('b2', [], initializer=init_b)
b3 = tf.get_variable('b3', [], initializer=init_b)

L1 = None
L2 = None

logits = None

cost = None
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
    print('3 fully-connected')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 2.01068e+00
# epoch:  2, cost: 6.26974e-01
# epoch:  3, cost: 3.83629e-01
# epoch:  4, cost: 2.51058e-01
# epoch:  5, cost: 1.59235e-01
# epoch:  6, cost: 1.01227e-01
# epoch:  7, cost: 6.48902e-02
# epoch:  8, cost: 3.95720e-02
# epoch:  9, cost: 2.46660e-02
# epoch: 10, cost: 1.59081e-02
# epoch: 11, cost: 8.53199e-03
# epoch: 12, cost: 5.35054e-03
# epoch: 13, cost: 3.00625e-03
# epoch: 14, cost: 2.08527e-03
# epoch: 15, cost: 1.57170e-03
# 3fc
# accuracy: 0.92810
