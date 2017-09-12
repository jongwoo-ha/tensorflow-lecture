import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

sample = 'hi hello'
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

seq_length = len(sample)-1
num_classes = len(idx2char)
learning_rate = 0.01
steps = 200

def encode(string, shape=[-1, 1]):
    return np.array([char2idx[c] for c in string if c in idx2char]).reshape(shape)

def decode(array):
    return ''.join([idx2char[i] for i in np.squeeze(array)])

seq = encode(sample)

X = tf.placeholder(tf.int32, [1])
Y = tf.placeholder(tf.int32, [1])
state = tf.placeholder(tf.float32, [1, num_classes])

X_one_hot = tf.one_hot(X, num_classes)
Y_one_hot = tf.one_hot(Y, num_classes)

cell = tf.contrib.rnn.BasicRNNCell(num_classes)
init_state = cell.zero_state(1, tf.float32)
logits, out_state = cell.call(X_one_hot, state)
prediction = tf.argmax(logits, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=logits))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        s = sess.run(init_state)
        cost_sum = 0
        for i in range(0, seq_length):
            s, c, _ = sess.run([out_state, cost, train], {X: seq[i], Y:seq[i+1], state: s})
            cost_sum += c
        if step % 20 == 0:
            print('step: %d, cost: %.5e' % (step, cost_sum/seq_length))
    
    predictions = []
    s = sess.run(init_state)
    for i in range(0, seq_length):
        p, s = sess.run([prediction, out_state], {X: seq[i], state: s})
        predictions.append(p)
    
    print(decode(predictions))
