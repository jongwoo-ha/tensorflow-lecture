import tensorflow as tf

x_train = [0, 1, 2, 3, 4, 5]
y_train = [1, 2, 3, 4, 5, 6]
x_test = 6

learning_rate = 0.05
steps = 2000

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(-1., tf.float32)
b = tf.Variable(-1., tf.float32)

h = w * x + b

cost = tf.reduce_mean(tf.square(h - y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps):
    _cost, _ = sess.run([cost, train], {x: x_train, y: y_train})
    if step % 100 == 0:
        print('step: %04d, cost: %.5e' % (step, _cost))


print('w: %.5f, b: %.5f, cost: %.5e' % tuple(sess.run([w, b, cost], {x: x_train, y: y_train})))
print('x: %.5f, y: %.5f' % (x_test, sess.run(h, {x: x_test})))
