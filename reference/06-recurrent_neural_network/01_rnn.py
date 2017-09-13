import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

sample = 'hi hello'
idx2char = list(set(sample))
char2idx = {_cost: i for i, _cost in enumerate(idx2char)}

seq_length = len(sample)-1
num_classes = len(idx2char)
num_units = num_classes
learning_rate = 0.01
steps = 500

def encode(string, shape=[-1, 1]):
    return np.array([char2idx[_cost] for _cost in string if _cost in idx2char]).reshape(shape)

def decode(array):
    return ''.join([idx2char[i] for i in np.reshape(array, [-1])])

seq = encode(sample)

X = tf.placeholder(tf.int32, [1])
Y = tf.placeholder(tf.int32, [1])
state = tf.placeholder(tf.float32, [1, num_units])

X_one_hot = tf.one_hot(X, num_classes) # (batch_size, input_dim) 
Y_one_hot = tf.one_hot(Y, num_classes) # (batch_size, output_dim)

cell = tf.contrib.rnn.BasicRNNCell(num_units)
init_state = cell.zero_state(1, tf.float32) # (batch_size, num_units)
outputs, out_state = cell.call(X_one_hot, state) # (batch_size, num_units)
prediction = tf.argmax(outputs, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=outputs))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(steps):
        _state = sess.run(init_state)
        cost_sum = 0
        for i in range(0, seq_length):
            _state, _cost, _ = sess.run([out_state, cost, train], {X: seq[i], Y:seq[i+1], state: _state})
            cost_sum += _cost
        if step % 100 == 0:
            print('step: %d, cost: %.5e' % (step, cost_sum/seq_length))
    
    inputs = []
    predictions = []
    _state = sess.run(init_state)
    for i in range(0, seq_length):
        p, _state = sess.run([prediction, out_state], {X: seq[i], state: _state})
        inputs.append(seq[i])
        predictions.append(p)
    
    print(decode(inputs))
    print(decode(predictions))
