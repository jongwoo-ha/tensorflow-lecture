import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

xy = np.loadtxt('../../dataset/01-test_score.csv', delimiter=',')
x_data = None
y_data = None

learning_rate = 3e-5
steps = 2000

X = tf.placeholder(tf.float32, shape=[])
Y = tf.placeholder(tf.float32, shape=[])

W = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

mae = None

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps):
    _cost, _ = sess.run([cost, train], {X: x_data, Y: y_data})
    if step < 10 or step % 100 == 0:
        print('step: %d, cost: %.5e' % (step, _cost))

print(sess.run(H, {X: x_data}))
print('MAE: %.5f' % sess.run(mae, {X: x_data, Y: y_data}))
