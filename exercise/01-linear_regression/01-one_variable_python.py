def h(x, w, b):
    return 0.

def cost(xs, w, b, ys):
    m = len(xs)
    sum = 0.
    for x, y in zip(xs, ys):
        pass
    return sum / (2*m)

def derivative_j(xs, w, b, ys):
    m = len(xs)
    sum_w, sum_b = 0., 0.
    for x, y in zip(xs, ys):
        pass
    return sum_w/m, sum_b/m

def gradient_descent(xs, w, b, ys, alpha):
    djw, djb = derivative_j(xs, w, b, ys)
    w = 0
    b = 0
    return w, b

x_train = [0, 1, 2, 3, 4, 5]
y_train = [1, 2, 3, 4, 5, 6]
x_test = 6

w, b = -1., -1.
learning_rate = 0.05
steps = 2000

for step in range(steps):
    w, b = gradient_descent(x_train, w, b, y_train, learning_rate)
    if step % 100 == 0:
        print('step: %4d, cost: %.5e' % (step, cost(x_train, w, b, y_train)))

print('w: %.5f, b: %.5f, cost: %.5e' % (w, b, cost(x_train, w, b, y_train)))
print('x: %.5f, y: %.5f' % (x_test, h(x_test, w, b)))
