import tensorflow as tf

x_train = [0, 1, 2, 3, 4, 5]
y_train = [1, 2, 3, 4, 5, 6]
x_test = 6

learning_rate = 0.05
steps = 2000

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(-1., tf.float32)
b = tf.Variable(-1., tf.float32)

m = w * x
h = m + b
e = h - y
s = tf.square(e)
j = s / 2
J = tf.reduce_mean(j)

dj_ds = tf.constant(1/2, tf.float32)
ds_de = 2*e
de_dh = tf.constant(1, tf.float32)
dh_dm = tf.constant(1, tf.float32)
dh_db = tf.constant(1, tf.float32)
dm_dw = x

dj_db = dj_ds * ds_de * de_dh * dh_db
dj_dw = dj_ds * ds_de * de_dh * dh_dm * dm_dw

dJ_db = tf.reduce_mean(dj_db, 0)
dJ_dw = tf.reduce_mean(dj_dw, 0)

train_w = w.assign_sub(learning_rate*dJ_dw)
train_b = b.assign_sub(learning_rate*dJ_db)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps):
    _cost, _, __ = sess.run([J, train_w, train_b], {x: x_train, y: y_train})
    if step % 100 == 0:
        print('step: %04d, cost: %.5e' % (step, _cost))


print('w: %.5f, b: %.5f, cost: %.5e' % tuple(sess.run([w, b, J], {x: x_train, y: y_train})))
print('x: %.5f, y: %.5f' % (x_test, sess.run(h, {x: x_test})))
