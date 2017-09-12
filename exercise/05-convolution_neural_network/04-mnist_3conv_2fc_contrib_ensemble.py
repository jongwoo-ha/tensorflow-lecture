import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, dropout, fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import datetime

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

learning_rate = 0.001
epochs = 15
batch_size = 100
keep = 0.75
num_models = 5

class Model:
    def __init__(self):
        self._build_net()
    
    def _build_net(self):
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.is_training = tf.placeholder_with_default(False, [])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        X_img = tf.reshape(self.X, [-1, 28, 28, 1])
        
        L1 = dropout(max_pool2d(conv2d(X_img, 32, [3, 3]), [2, 2], padding='SAME'), keep, is_training=self.is_training)
        L2 = dropout(max_pool2d(conv2d(L1, 64, [3, 3]), [2, 2], padding='SAME'), keep, is_training=self.is_training)
        L3 = dropout(max_pool2d(conv2d(L2, 128, [3, 3]), [2, 2], padding='SAME'), keep, is_training=self.is_training)
        L3 = tf.reshape(L3, [-1, 4*4*128])
        L4 = dropout(fully_connected(L3, 512), keep, is_training=self.is_training)
        self.logits = fully_connected(L4, 10, activation_fn=None)
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits))
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1)), tf.float32))
        
    def train(self, sess, x_train, y_train):
        return sess.run([self.cost, self.optimize], {self.X: x_train, self.Y: y_train, self.is_training: True})
    
    def get_logits(self, sess, x_test):
        return sess.run(self.logits, {self.X: x_test})

    def get_accuracy(self, sess, x_test, y_test):
        return sess.run(self.accuracy, {self.X: x_test, self.Y: y_test})

models = [Model() for i in range(num_models)]

with tf.Session() as sess:
    start = datetime.datetime.now()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        cost_sum = 0
        num_batches = int(mnist.train.num_examples / batch_size)
     
        for i in range(num_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
             
            for m in models:
                c, _ = m.train(sess, x_batch, y_batch)
                cost_sum += c
         
        print('epoch: %2d, cost: %.5e' % ((epoch+1), cost_sum/(num_batches*num_models)))
    print('train:', datetime.datetime.now()-start)
    
    start = datetime.datetime.now()
    print('3-conv, 2-fc, %d-ensemble, contrib' % num_models)
    test_size = len(mnist.test.labels)
    sum_logits = np.zeros([test_size, 10], np.float32)
    for m_idx, model in enumerate(models):
        print('model: %d, accuracy: %.5f' % (m_idx, model.get_accuracy(sess, mnist.test.images, mnist.test.labels)))
        logits = model.get_logits(sess, mnist.test.images)
        sum_logits += logits
    
    is_correct = tf.equal(tf.argmax(sum_logits, 1), tf.argmax(mnist.test.labels, 1))
    ensemble_accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('ensemble accuracy: %.5f' % sess.run(ensemble_accuracy))
    print('test:', datetime.datetime.now()-start)

# epoch:  1, cost: 2.40237e-01
# epoch:  2, cost: 7.11121e-02
# epoch:  3, cost: 5.44674e-02
# epoch:  4, cost: 4.48269e-02
# epoch:  5, cost: 3.94313e-02
# epoch:  6, cost: 3.52247e-02
# epoch:  7, cost: 3.10722e-02
# epoch:  8, cost: 2.86143e-02
# epoch:  9, cost: 2.61258e-02
# epoch: 10, cost: 2.53168e-02
# epoch: 11, cost: 2.31011e-02
# epoch: 12, cost: 2.17428e-02
# epoch: 13, cost: 2.02977e-02
# epoch: 14, cost: 1.89458e-02
# epoch: 15, cost: 1.81220e-02
# model: 0, accuracy: 0.99330
# model: 1, accuracy: 0.99360
# model: 2, accuracy: 0.99390
# model: 3, accuracy: 0.99340
# model: 4, accuracy: 0.99420
# Ensemble accuracy: 0.9948
