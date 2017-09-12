import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

train_ratio = 0.8
learning_rate = 0.2
num_steps = 5000
epsilon = 1e-12

xy = np.loadtxt('../../dataset/02-stock_google.csv', delimiter=',')
train_size = int((xy.shape[0] - 1) * train_ratio)
x_train, y_train = xy[:train_size], xy[1:train_size+1, [-1]]
x_test, y_test = xy[train_size:-1], xy[train_size+1:, [-1]]
num_features = x_train.shape[1]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

train_data = np.hstack((x_train, y_train))
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)

X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None, 1])
M = tf.placeholder(tf.float32, [num_features+1])
S = tf.placeholder(tf.float32, [num_features+1])

W = tf.Variable(tf.random_normal([num_features, 1]))
b = tf.Variable(tf.random_normal([1]))

X_mean, Y_mean = M[:num_features], M[num_features:]
X_std, Y_std = S[:num_features], S[num_features:]

X_norm = (X - X_mean) / (X_std + epsilon)
Y_norm = (Y - Y_mean) / (Y_std + epsilon)

H_norm = tf.matmul(X_norm, W) + b

cost = tf.reduce_mean(tf.square(H_norm - Y_norm))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction = H_norm * (Y_std + epsilon) + Y_mean

abs_diff = tf.abs(prediction - Y)
mae, min_diff, max_diff =  tf.reduce_mean(abs_diff), tf.reduce_min(abs_diff), tf.reduce_max(abs_diff)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(num_steps):
        _cost, _mae, _ = sess.run([cost, mae, train], {X: x_train, Y: y_train, M: mean, S: std})
        if step < 10 or step % 100 == 0:
            print('step: %4d, cost: %.5e, mae: %.4f' % (step, _cost, _mae))
    
    print('MAE: %.4f, MinDiff: %.4f, MaxDiff: %.4f' % tuple(sess.run([mae, min_diff, max_diff], {X: x_test, Y: y_test, M: mean, S: std})))
