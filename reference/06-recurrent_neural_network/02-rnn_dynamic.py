import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

sample = 'hi hello'
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

seq_length = len(sample)-1
num_classes = len(idx2char)
input_dim = output_dim = num_units = num_classes
learning_rate = 0.01
steps = 500

def encode(string, shape=[1, -1]):
    return np.array([char2idx[c] for c in string if c in idx2char]).reshape(shape)

def decode(array):
    return ''.join([idx2char[i] for i in np.reshape(array, [-1])])

x_data = encode(sample[:-1]) # 'hi hell' (1, 7)
y_data = encode(sample[1:])  # 'i hello'  (1, 7)

X = tf.placeholder(tf.int32, [None, None])
Y = tf.placeholder(tf.int32, [None, None])

X_one_hot = tf.one_hot(X, input_dim) # (batch_size, seq_length, input_dim)
Y_one_hot = tf.one_hot(Y, output_dim) # (batch_size, seq_length, output_dim)

cell = tf.contrib.rnn.BasicRNNCell(num_units)
outputs, out_state = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
prediction = tf.argmax(outputs, 2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=outputs))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _cost, _prediction, _ = sess.run([cost, prediction, train], {X: x_data, Y: y_data})
        if step % 100 == 0:
            print('step: %d, cost: %.5e, prediction: %s' % (step, _cost, decode(_prediction)))
    
    input = 'h'
    for _ in range(seq_length):
        p = decode(sess.run(prediction, {X: encode(input)}))
        print('input: %s, prediction: %s' % (input, p))
        input += p[-1:]
    print(input)
