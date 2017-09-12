import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

xy = np.loadtxt('../../dataset/04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, -1:]

learning_rate = 0.1
steps = 1000
num_features = x_data.shape[1]
num_classes = 7

X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.reshape(tf.one_hot(Y, num_classes), [-1, num_classes])

W = tf.get_variable('W', [num_features, num_classes])
b = tf.get_variable('b', [num_classes])

logits = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=logits))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y_one_hot, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _cost, _ = sess.run([cost, train], {X: x_data, Y: y_data})
        if step < 20 or step % 100 == 0:
            print('step: %d, cost: %.5e' % (step, _cost))
        
    print('accuracy: %.5f' % sess.run(accuracy, {X: x_data, Y: y_data}))
