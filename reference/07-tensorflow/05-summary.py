import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

def variable_summaries(var, name):
    with tf.name_scope('summaries_'+name):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

X = tf.placeholder(tf.float32, [None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')

init_w = tf.random_normal_initializer()
init_b = tf.zeros_initializer()
W = tf.get_variable('W', [784, 10], initializer=init_w)
b = tf.get_variable('b', [10], initializer=init_b)

logits = tf.add(tf.matmul(X, W), b)
xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
prediction = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 1)), tf.float32))
cost = tf.reduce_mean(xentropy)
train = tf.train.AdamOptimizer(0.001).minimize(cost)

variable_summaries(W, 'weight')
variable_summaries(b, 'bias')
tf.summary.image('input', tf.reshape(X, [-1, 28, 28, 1]), 10)
tf.summary.histogram('logits', logits)
summ_xentropy = tf.summary.histogram('cross entropy', xentropy)
summ_cost = tf.summary.scalar('cost', cost)
summ_accuracy = tf.summary.scalar('accuracy', accuracy)

summ_all = tf.summary.merge_all()
summ_test = tf.summary.merge([summ_cost, summ_accuracy, summ_xentropy])

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('temp/05-summary/train', sess.graph)
    test_writer = tf.summary.FileWriter('temp/05-summary/test')
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(5000):
        x_batch, y_batch = mnist.train.next_batch(100)

        if (step+1) % 1000 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _summ, _cost, _ = sess.run(fetches=[summ_all, cost, train],
                                       feed_dict={X: x_batch, Y: y_batch},
                                       options=run_options,
                                       run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step_%6d' % (step+1))
            train_writer.add_summary(_summ, (step+1))
            
            _summ = sess.run(summ_test, {X: mnist.test.images, Y: mnist.test.labels})
            test_writer.add_summary(_summ, (step+1))
            
            print('step: %d, cost: %.5e' % ((step+1), _cost))
        else:
            _summ, _cost, _ = sess.run([summ_all, cost, train], {X: x_batch, Y: y_batch})
            train_writer.add_summary(_summ, (step+1))
    
    print('softmax')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))

# tensorboard --logdir=temp/05-summary
