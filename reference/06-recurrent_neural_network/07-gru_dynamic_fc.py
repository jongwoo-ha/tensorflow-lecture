# https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/11_char_rnn_gist.py
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/

import tensorflow as tf

tf.set_random_seed(0)

vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "\\^_abcdefghijklmnopqrstuvwxyz{|}")
DATA_PATH = '../../dataset/05_arvix_abstracts.txt'
HIDDEN_SIZE = 200
BATCH_SIZE = 64
NUM_STEPS = 50
SKIP_STEP = 100
TEMPRATURE = 0.7
LR = 0.003
LEN_GENERATED = 300

def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window=NUM_STEPS, overlap=NUM_STEPS//2):
    for text in open(filename):
        text = vocab_encode(text, vocab)
        for start in range(0, len(text) - overlap, overlap):
            chunk = text[start: start + window]
            chunk += [0] * (window - len(chunk))
            yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

seq = tf.placeholder(tf.int32, [None, None])
temp = tf.placeholder(tf.float32)

seq_one_hot = tf.one_hot(seq, len(vocab))

cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)

in_state = tf.placeholder_with_default(cell.zero_state(tf.shape(seq_one_hot)[0], tf.float32), [None, HIDDEN_SIZE])
length = tf.reduce_sum(tf.reduce_max(tf.sign(seq_one_hot), 2), 1)
output, out_state = tf.nn.dynamic_rnn(cell, seq_one_hot, length, in_state)

logits = tf.contrib.layers.fully_connected(output, len(vocab), None)

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=seq_one_hot[:, 1:]))

sample = tf.multinomial(tf.exp(logits[:, -1] / temp), 1)[:, 0]

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(LR).minimize(cost)

def online_inference(sess):
    sentence = 'T'
    state = None
    for _ in range(LEN_GENERATED):
        batch = [vocab_encode(sentence[-1], vocab)]
        feed = {seq: batch, temp: TEMPRATURE}
        if state is not None:
            feed[in_state] = state
        index, state = sess.run([sample, out_state], feed)
        sentence += vocab_decode(index, vocab)
    print(sentence)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step, batch in enumerate(read_batch(read_data(DATA_PATH, vocab))):
        batch_loss, _ = sess.run([cost, optimizer], {seq: batch})
        if step % SKIP_STEP == 0:
            print('step: %d, cost: %.5e' % (step, batch_loss))
            online_inference(sess)
    online_inference(sess)
