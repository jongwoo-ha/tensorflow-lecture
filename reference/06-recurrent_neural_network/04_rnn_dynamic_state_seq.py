import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

np.set_printoptions(formatter={'all':lambda x: '%+.2f' % x})

sample = 'hi hello'
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

seq_length = len(sample)-1
num_classes = len(idx2char)
input_dim = output_dim = num_units = num_classes
learning_rate = 0.01
steps = 200

def encode(string, shape=[1, -1]):
    return np.array([char2idx[c] for c in string if c in idx2char]).reshape(shape)

def decode(array):
    return ''.join([idx2char[i] for i in np.reshape(array, [-1])])

sequence = encode(sample)

S = tf.placeholder(tf.int32, [None, None])
S_one_hot = tf.one_hot(S, num_classes)

cell = tf.contrib.rnn.BasicRNNCell(num_units)
init_state = tf.placeholder_with_default(cell.zero_state(tf.shape(S)[0], tf.float32), [None, num_units])
outputs, out_state = tf.nn.dynamic_rnn(cell, S_one_hot, initial_state=init_state, dtype=tf.float32)
prediction = tf.argmax(outputs, 2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=S_one_hot[:, 1:], logits=outputs[:, :-1]))
# weights = tf.ones(tf.shape(S))[:, :-1]
# cost = tf.contrib.seq2seq.sequence_loss(logits=outputs[:, :-1], targets=S[:, 1:], weights=weights)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _cost, _prediction, _ = sess.run([cost, prediction, train], {S: sequence})
        if step % 20 == 0:
            print('step: %d, cost: %.5e, prediction: %s' % (step, _cost, decode(_prediction[:, :-1])))
    
    print('')
    input = 'h'
    for _ in range(seq_length):
        p = decode(sess.run(prediction, {S: encode(input)}))
        print('input: %s, prediction: %s' % (input, p))
        input += p[-1:]
    print(input)
    
    print('')
    sentence = input = 'h'
    _init = sess.run(init_state, {S: encode(input)})
    for _ in range(seq_length):
        p, _out = sess.run([prediction, out_state], {S: encode(input), init_state: _init})
        print('in: %s, input: %s, prediction: %s, out: %s' % (_init, input, decode(p), _out))
        _init, input = _out, decode(p)
        sentence += input
    print(sentence)
