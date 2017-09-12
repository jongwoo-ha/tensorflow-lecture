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

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), tf.float32)
b = tf.Variable(tf.random_normal([1]), tf.float32)

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps):
    _cost, _ = sess.run([cost, train], {X: x_data, Y: y_data})
    if step < 10 or step % 100 == 0:
        print('step: %d, cost: %.5e' % (step, _cost))

print(sess.run(H, {X: x_data}))
