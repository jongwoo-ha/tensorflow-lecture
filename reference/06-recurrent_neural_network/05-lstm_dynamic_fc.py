import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

sample = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

seq_length = 64
num_classes = len(idx2char)
input_dim = output_dim = num_classes
num_units = 128
learning_rate = 0.1
steps = 1000

def encode(string, shape=[-1]):
    return np.array([char2idx[c] for c in string if c in idx2char]).reshape(shape)

def decode(array):
    return ''.join([idx2char[i] for i in np.reshape(array, [-1])])

sequences = []
for i in range(0, len(sample) - seq_length + 1):
    sequences.append(encode(sample[i : i+seq_length]))
sequences = np.array(sequences)

print(len(sequences))
    
S = tf.placeholder(tf.int32, [None, None])
S_one_hot = tf.one_hot(S, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units)
outputs, _state = tf.nn.dynamic_rnn(cell, S_one_hot, dtype=tf.float32)
logits = tf.contrib.layers.fully_connected(outputs, output_dim, None)
prediction = tf.argmax(logits, 2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=S_one_hot[:, 1:]))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        c, _ = sess.run([cost, train], {S: sequences})
        if step < 10 or step % 100 == 0:
            print('step: %d, cost: %.5e' % (step, c))
    
    result = sess.run(prediction, {S: sequences[:, :-1]})
    result = decode(sequences[0])[:-1] + decode(result[:, [-1]])
    print('sample:', sample)
    print('result:', result)
    print('sample equals result:', result == sample)
    
    input = 'i'
    for i in range(len(sample)-1):
        p = decode(sess.run(prediction, {S: encode(input).reshape([-1, len(input)])}))
        input += p[-1:]
    print('generate:', input)
    
    
