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

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable('W1', [2, 2])
b1 = tf.get_variable('b1', [2])

W2 = tf.get_variable('W2', [2, 1])
b2 = tf.get_variable('b2', [1])

L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

logits = tf.matmul(L1, W2) + b2
H = tf.sigmoid(logits)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)) 
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
