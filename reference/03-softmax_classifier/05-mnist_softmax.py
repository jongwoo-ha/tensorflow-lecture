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

W = tf.get_variable('W', [784, 10], initializer=init_w)
b = tf.get_variable('b', [10], initializer=init_b)

logits = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
# train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
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
    print('softmax')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 5.64335e+00
# epoch:  2, cost: 1.77278e+00
# epoch:  3, cost: 1.15695e+00
# epoch:  4, cost: 9.18546e-01
# epoch:  5, cost: 7.86035e-01
# epoch:  6, cost: 6.97829e-01
# epoch:  7, cost: 6.34407e-01
# epoch:  8, cost: 5.86185e-01
# epoch:  9, cost: 5.47761e-01
# epoch: 10, cost: 5.16630e-01
# epoch: 11, cost: 4.91037e-01
# epoch: 12, cost: 4.68377e-01
# epoch: 13, cost: 4.49647e-01
# epoch: 14, cost: 4.32903e-01
# epoch: 15, cost: 4.19361e-01
# softmax
# accuracy: 0.89870
