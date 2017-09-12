import tensorflow as tf

x_train = [0, 1, 2, 3, 4, 5]
y_train = [1, 2, 3, 4, 5, 6]
x_test = 6

learning_rate = 0.05
steps = 2000

x = None
y = None

w = None
b = None

h = None

cost = None

train = None

sess = None
sess.run(None)

for step in range(steps):
    _cost, _ = sess.run([cost, train], {None})
    if step % 100 == 0:
        print('step: %04d, cost: %.5e' % (step, _cost))


print('w: %.5f, b: %.5f, cost: %.5e' % tuple(sess.run([w, b, cost], {None})))
print('x: %.5f, y: %.5f' % (x_test, sess.run(h, {None})))
