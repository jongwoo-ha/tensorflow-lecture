import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

train_ratio = 0.75
steps = 2000
learning_rate = 0.5

xy = np.loadtxt('../../dataset/03-diabetes.csv', delimiter=',', dtype=np.float32)
train_size = int(xy.shape[0]*train_ratio)
x_train, y_train = xy[:train_size, 0:-1], xy[:train_size, -1:]
x_test, y_test = xy[train_size:, 0:-1], xy[train_size:, -1:]
num_features = x_train.shape[1]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None, 1])
threshold = tf.placeholder_with_default(0.5, [])

W = tf.Variable(tf.random_normal([num_features, 1]))
b = tf.Variable(tf.random_normal([1]))

logits = tf.matmul(X, W) + b
H = tf.sigmoid(logits)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)) 
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

P = tf.cast(H > threshold, dtype=tf.float32)

tp = tf.count_nonzero(P * Y)
tn = tf.count_nonzero((P - 1) * (Y - 1))
fp = tf.count_nonzero(P * (Y - 1))
fn = tf.count_nonzero((P - 1) * Y)

accuracy_ = tf.reduce_mean(tf.cast(tf.equal(P, Y), tf.float32))
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):
        _cost, _ = sess.run([cost, train], {X: x_train, Y: y_train})
        if step < 20 or step % 100 == 0:
            print('step: %d, cost: %.5e' % (step, _cost))
            
    print('threshold: %.2f, accuracy_: %.5f, accuracy: %.5f' %
          tuple(sess.run([threshold, accuracy_, accuracy], {X: x_test, Y: y_test})))
    
    for t in np.arange(0.1, 1.0, 0.1):
        print('[%.1f]' % t, 'accuracy: %.5f, precision: %.5f, recall: %.5f, f1: %.5f' %
          tuple(sess.run([accuracy, precision, recall, f1], {X: x_test, Y: y_test, threshold: t})))
    
# step: 1900, cost: 4.79366e-01
# threshold: 0.50, accuracy: 0.77895
# [0.1] accuracy: 0.64737, precision: 0.64324, recall: 0.99167, f1: 0.78033
# [0.2] accuracy: 0.69474, precision: 0.67816, recall: 0.98333, f1: 0.80272
# [0.3] accuracy: 0.73158, precision: 0.71166, recall: 0.96667, f1: 0.81979
# [0.4] accuracy: 0.77895, precision: 0.76351, recall: 0.94167, f1: 0.84328
# [0.5] accuracy: 0.77895, precision: 0.77465, recall: 0.91667, f1: 0.83969
# [0.6] accuracy: 0.77368, precision: 0.80800, recall: 0.84167, f1: 0.82449
# [0.7] accuracy: 0.80000, precision: 0.89423, recall: 0.77500, f1: 0.83036
# [0.8] accuracy: 0.66842, precision: 0.93846, recall: 0.50833, f1: 0.65946
# [0.9] accuracy: 0.52632, precision: 1.00000, recall: 0.25000, f1: 0.40000
