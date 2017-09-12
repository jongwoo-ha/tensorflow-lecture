# scalar for each feature and parameter

import tensorflow as tf

x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]

learning_rate = 3e-5 # 3e-5 == 0.00003
steps = 2000

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(0., tf.float32)
w2 = tf.Variable(0., tf.float32)
w3 = tf.Variable(0., tf.float32)
b = tf.Variable(0., tf.float32)

H = None

cost = tf.reduce_mean(tf.square(H - Y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps):
    _cost, _ = sess.run([cost, train], {None})
    if step < 10 or step % 100 == 0:
        print('step: %d, cost: %.5e' % (step, _cost))

print(sess.run(H, {None}))

# step: 9900, cost: 1.62855e+00
# [ 150.43339539  185.1410675   180.09382629  197.94418335  140.6676178 ]