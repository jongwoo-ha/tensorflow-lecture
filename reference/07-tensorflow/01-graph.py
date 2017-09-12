import tensorflow as tf

tf.set_random_seed(0)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

init_w = tf.random_normal_initializer()
init_b = tf.zeros_initializer()

W = tf.get_variable('W', [784, 10], initializer=init_w)
b = tf.get_variable('b', [10], initializer=init_b)

logits = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(cost)

prediction = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 1)), tf.float32))

initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('graph/01_graph', sess.graph)
    writer.close()
    print('done')

# tensorboard --logdir=graph
