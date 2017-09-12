import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

learning_rate = 0.001
epochs = 15
batch_size = 100
keep = 0.75

X = tf.placeholder(tf.float32, [])
Y = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder_with_default(1.0, None)

init_w = tf.contrib.layers.xavier_initializer()
init_b = tf.zeros_initializer()

conv_W1 = tf.get_variable('conv_W1', [], initializer=init_w)
conv_W2 = tf.get_variable('conv_W2', [], initializer=init_w)

fc_W1 = tf.get_variable('fc_W1', [], initializer=init_w)
fc_b1 = tf.get_variable('fc_b1', [], initializer=init_b)

X_img = tf.reshape(X, [])

L1 = tf.nn.conv2d(X_img, conv_W1, [], 'SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, [], [], 'SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

L2 = tf.nn.conv2d(L1, conv_W2, [], 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, [], [], 'SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L2 = tf.reshape(L2, [])

logits = tf.matmul(L2, fc_W1) + fc_b1

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
    print('2-conv, 1-fc')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 2.98218e-01
# epoch:  2, cost: 9.42262e-02
# epoch:  3, cost: 7.32832e-02
# epoch:  4, cost: 6.04302e-02
# epoch:  5, cost: 5.17074e-02
# epoch:  6, cost: 4.81039e-02
# epoch:  7, cost: 4.14954e-02
# epoch:  8, cost: 3.89885e-02
# epoch:  9, cost: 3.70210e-02
# epoch: 10, cost: 3.49232e-02
# epoch: 11, cost: 3.12615e-02
# epoch: 12, cost: 2.75115e-02
# epoch: 13, cost: 2.76558e-02
# epoch: 14, cost: 2.57930e-02
# epoch: 15, cost: 2.43807e-02
# 2-conv, 1-fc
# accuracy: 0.99120
