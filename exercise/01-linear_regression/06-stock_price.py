import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

train_ratio = 0.8
learning_rate = 1e-14 
steps = 5000

xy = np.loadtxt('../../dataset/02-stock_google.csv', delimiter=',')
train_size = int((xy.shape[0] - 1) * train_ratio)
x_train, y_train = None, None
x_test, y_test = None, None
num_features = None
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

X = tf.placeholder(tf.float32, [])
Y = tf.placeholder(tf.float32, [])

W = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

abs_diff = tf.abs(H - Y)
mae, min_diff, max_diff =  None, None, None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _cost, _ = sess.run([cost, train], {X: x_train, Y: y_train})
        if step < 10 or step % 100 == 0:
            print('step: %4d, cost: %.5e, mae: %.4f' % (step, _cost, sess.run(mae, {X: x_test, Y: y_test})))
    
    print('MAE: %.4f, MinDiff: %.4f, MaxDiff: %.4f' % tuple(sess.run([mae, min_diff, max_diff], {X: x_test, Y: y_test})))

# xy = xy[:, [0,1,2,4]]
# 1e-7
