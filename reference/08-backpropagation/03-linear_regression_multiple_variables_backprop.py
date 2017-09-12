import tensorflow as tf

tf.set_random_seed(0)

x_data = [[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]
y_data = [[152],
          [185],
          [180],
          [196],
          [142]]

learning_rate = 3e-5
steps = 2000

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), tf.float32)
b = tf.Variable(tf.random_normal([1]), tf.float32)

M = tf.matmul(X, W) + b
H = M + b
E = H - Y
S = tf.square(E)
j = 1/2*S
J = tf.reduce_mean(j)

dj_dS = tf.constant(1/2, tf.float32)
dS_dE = tf.constant(2*E, tf.float32)
dE_dH = tf.constant(1, tf.float32)
dH_dM = tf.constant(1, tf.float32)
dH_db = tf.constant(1, tf.float32)
dM_dW = X

dj_dW = dj_dS * dS_dE * dE_dH * dH_dM * dM_dW
dj_db = dj_dS * dS_dE * dE_dH * dH_db

dJ_dW = tf.reshape(tf.reduce_mean(dj_dW, 0), tf.shape(W))
dJ_db = tf.reshape(tf.reduce_mean(dj_db, 0), tf.shape(b))

train_W = W.assign_sub(learning_rate * dJ_dW)
train_b = b.assign_sub(learning_rate * dJ_db)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps):
    _cost, _, __ = sess.run([J, train_W, train_b], {X: x_data, Y: y_data})
    if step < 10 or step % 100 == 0:
        print('step: %d, cost: %.5e' % (step, _cost))

print(sess.run(H, {X: x_data}))
