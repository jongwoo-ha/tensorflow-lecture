import tensorflow as tf

tf.set_random_seed(0)

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]

steps = 2000
learning_rate = 0.1

X = tf.placeholder(tf.float32, shape=[])
Y = tf.placeholder(tf.float32, shape=[])

W1 = tf.get_variable('W1', [])
b1 = tf.get_variable('b1', [])

W2 = tf.get_variable('W2', [])
b2 = tf.get_variable('b2', [])

L1 = None

logits = None
H = None

cost = None 
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _cost, _ = sess.run([cost, train], {X: x_data, Y: y_data})
        if step < 20 or step % 100 == 0:
            print('step: %d, cost: %.5e' % (step, _cost))
    
    print(sess.run(prediction, {X: x_data}))
    print('accuracy:', sess.run(accuracy, {X: x_data, Y: y_data}))
