import tensorflow as tf

tf.set_random_seed(0)

x_data = [[1, 2, 1, 1],
          [1, 1, 2, 1],
          [1, 1, 3, 1],
          [1, 2, 4, 3],
          [1, 3, 3, 3],
          [1, 4, 3, 2],
          [1, 6, 6, 7],
          [1, 7, 7, 8],
          [1, 8, 7, 9]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1],
          [2],
          [2],
          [2]]

learning_rate = 0.1
steps = 1000

X = tf.placeholder(tf.float32, [])
Y = tf.placeholder(tf.int32, [])

Y_one_hot = None

W = tf.get_variable('W', [])
b = tf.get_variable('b', [])

logits = tf.matmul(X, W) + b

cost = None
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(logits, 1)
accuracy = None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _cost, _ = sess.run([cost, train], {X: x_data, Y: y_data})
        if step < 20 or step % 100 == 0:
            print('step: %d, cost: %.5e' % (step, _cost))
        
    print(sess.run(logits, {X: x_data}))
    print(sess.run(prediction, {X: x_data}))
    print('accuracy: %.5f' % sess.run(accuracy, {X: x_data, Y: y_data}))
