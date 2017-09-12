import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

train_ratio = 0.8
learning_rate = 1e-14 
steps = 5000

xy = np.loadtxt('../../dataset/02-stock_google.csv', delimiter=',')
train_size = int((xy.shape[0] - 1) * train_ratio)
x_train, y_train = xy[:train_size], xy[1:train_size+1, [-1]]
x_test, y_test = xy[train_size:-1], xy[train_size+1:, [-1]]
num_features = x_train.shape[1]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([num_features, 1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

abs_diff = tf.abs(H - Y)
mae, min_diff, max_diff =  tf.reduce_mean(abs_diff), tf.reduce_min(abs_diff), tf.reduce_max(abs_diff)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _cost, _ = sess.run([cost, train], {X: x_train, Y: y_train})
        if step < 10 or step % 100 == 0:
            print('step: %4d, cost: %.5e, mae: %.4f' % (step, _cost, sess.run(mae, {X: x_test, Y: y_test})))
    
    print('MAE: %.4f, MinDiff: %.4f, MaxDiff: %.4f' % tuple(sess.run([mae, min_diff, max_diff], {X: x_test, Y: y_test})))

# xy = xy[:, [0,1,2,4]]
# 1e-7
