import tensorflow as tf

tf.set_random_seed(0)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

steps = 1000
learning_rate = 0.1

X = tf.placeholder(tf.float32, shape=[])
Y = tf.placeholder(tf.float32, shape=[])

W = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))

logits = None
H = None

cost = None 
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction = None
accuracy = None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):
        _cost, _ = sess.run([cost, train], {X: x_data, Y: y_data})
        if step < 20 or step % 100 == 0:
            print('step: %d, cost: %.5e' % (step, _cost))

    print(sess.run(prediction, {X: x_data, Y: y_data}))
