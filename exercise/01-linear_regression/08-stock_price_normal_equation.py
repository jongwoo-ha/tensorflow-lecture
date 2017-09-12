import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

train_ratio = 0.8

def normal_equation(x_train, y_train):
    X = np.hstack((np.ones([x_train.shape[0], 1]), x_train))
    XT = np.transpose(X)
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(XT, X)), XT), y_train)

xy = np.loadtxt('../../dataset/02-stock_google.csv', delimiter=',')
train_size = int((xy.shape[0] - 1) * train_ratio)
x_train, y_train = xy[:train_size], xy[1:train_size+1, [-1]]
x_test, y_test = xy[train_size:-1], xy[train_size+1:, [-1]]
num_features = x_train.shape[1]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

theta = normal_equation(x_train, y_train)

X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(theta[1:], dtype=tf.float32)
b = tf.Variable(theta[0:1].reshape([1]), dtype=tf.float32)

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))

abs_diff = tf.abs(H - Y)
mae, min_diff, max_diff =  tf.reduce_mean(abs_diff), tf.reduce_min(abs_diff), tf.reduce_max(abs_diff)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print('MAE: %.4f, MinDiff: %.4f, MaxDiff: %.4f' % tuple(sess.run([mae, min_diff, max_diff], {X: x_test, Y: y_test})))
