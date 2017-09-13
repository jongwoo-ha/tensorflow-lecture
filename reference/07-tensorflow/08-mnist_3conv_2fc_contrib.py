import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, dropout, fully_connected
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder_with_default(False, [])
X_img = tf.reshape(X, [-1, 28, 28, 1])
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

L1 = dropout(max_pool2d(conv2d(X_img, 32, [3, 3]), [2, 2], padding='SAME'), 0.75, is_training=is_training)
L2 = dropout(max_pool2d(conv2d(L1, 64, [3, 3]), [2, 2], padding='SAME'), 0.75, is_training=is_training)
L3 = dropout(max_pool2d(conv2d(L2, 128, [3, 3]), [2, 2], padding='SAME'), 0.75, is_training=is_training)
L3 = tf.reshape(L3, [-1, 4*4*128])
L4 = dropout(fully_connected(L3, 512), 0.75, is_training=is_training)
logits = fully_connected(L4, 10, activation_fn=None)
prediction = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 1)), tf.float32))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
train = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)

tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
summ_all = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('mnist/3conv_2fc/train', sess.graph)
    test_writer = tf.summary.FileWriter('mnist/3conv_2fc/test')
    
    ckpt = tf.train.get_checkpoint_state('mnist/3conv_2fc/checkpoints')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restore parameters (global_step=%d)' % global_step.eval())
    else:
        sess.run(tf.global_variables_initializer())
        
    for step in range(global_step.eval(), 10000):
        print('start training from %d steps' % (step+1))
        x_batch, y_batch = mnist.train.next_batch(100)

        if (step+1) % 1000 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _summ, _cost, _ = sess.run(fetches=[summ_all, cost,train],
                                       feed_dict={X: x_batch, Y: y_batch},
                                       options=run_options,
                                       run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step_%6d' % (step+1))
            train_writer.add_summary(_summ, (step+1))
            
            _summ = sess.run(summ_all, {X: mnist.test.images, Y: mnist.test.labels})
            test_writer.add_summary(_summ, (step+1))
            
            saver.save(sess, 'mnist/3conv_2fc/checkpoints/model', (step+1))
            print('step: %d, cost: %.5e' % ((step+1), _cost))
        else:
            _summ, _cost, _ = sess.run([summ_all, cost, train], {X: x_batch, Y: y_batch})
            if (step+1) % 10 == 0:
                train_writer.add_summary(_summ, (step+1))
    
    print('3-conv, 2-fc, contrib')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))

# tensorboard --logdir=mnist
