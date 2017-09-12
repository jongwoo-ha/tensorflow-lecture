import tensorflow as tf

tf.set_random_seed(0)

x_data = [[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]
y_data = [[152],
          [185],
          [180],
          [196],
          [142]]

learning_rate = 3e-5
steps = 2000

X = tf.placeholder(tf.float32, shape=[])
Y = tf.placeholder(tf.float32, shape=[])

W = tf.Variable(tf.random_normal([]), tf.float32)
b = tf.Variable(tf.random_normal([]), tf.float32)

H = None

cost = None

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps):
    _cost, _ = sess.run([cost, train], {None})
    if step < 10 or step % 100 == 0:
        print('step: %d, cost: %.5e' % (step, _cost))

print(sess.run(H, {None}))
