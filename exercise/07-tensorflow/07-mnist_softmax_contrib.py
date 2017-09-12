import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
init_w = tf.random_normal_initializer()
init_b = tf.zeros_initializer()
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

logits = tf.contrib.layers.fully_connected(X, 10, None, weights_initializer=init_w, biases_initializer=init_b)
prediction = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 1)), tf.float32))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
train = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)

tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
summ_all = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('mnist/softmax/train', sess.graph)
    test_writer = tf.summary.FileWriter('mnist/softmax/test')
    
    ckpt = tf.train.get_checkpoint_state('mnist/softmax/checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    print('start training from %d steps' % global_step.eval())
    for step in range(global_step.eval(), 10000):
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
            
            _summ = sess.run(summ_all, {X: mnist.test.images, Y: mnist.test.labels})
            test_writer.add_summary(_summ, (step+1))
            
            saver.save(sess, 'mnist/softmax/checkpoint/model', (step+1))
            print('step: %d, cost: %.5e' % ((step+1), _cost))
        else:
            _summ, _cost, _ = sess.run([summ_all, cost, train], {X: x_batch, Y: y_batch})
            if (step+1) % 10 == 0:
                train_writer.add_summary(_summ, (step+1))
    
    print('softmax')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))

# tensorboard --logdir=mnist
