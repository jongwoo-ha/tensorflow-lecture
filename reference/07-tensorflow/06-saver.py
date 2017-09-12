import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')

init_w = tf.random_normal_initializer()
init_b = tf.zeros_initializer()
W = tf.get_variable('W', [784, 10], initializer=init_w)
b = tf.get_variable('b', [10], initializer=init_b)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

logits = tf.add(tf.matmul(X, W), b)
xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
prediction = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y, 1)), tf.float32))
cost = tf.reduce_mean(xentropy)
train = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('temp/06-saver')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    print('start training from %d steps' % global_step.eval())
    for step in range(global_step.eval(), 5000):
        x_batch, y_batch = mnist.train.next_batch(100)
        
        _cost, _ = sess.run([cost, train], {X: x_batch, Y: y_batch})
        
        if (step+1) % 1000 == 0:
            saver.save(sess, 'temp/06-saver/model', (step+1))
            print('step: %d, cost: %.5e' % ((step+1), _cost))
    
    print('softmax')
    print('accuracy: %.5f' % sess.run(accuracy, {X: mnist.test.images, Y: mnist.test.labels}))

# tensorboard --logdir=temp/05-summary
