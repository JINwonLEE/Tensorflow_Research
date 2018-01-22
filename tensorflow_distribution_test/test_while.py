import tensorflow as tf



a, b = tf.while_loop(lambda a, b: a < 30,
    lambda a, b: (a * 3, b * 2),
    (2, 3))

result = tf.Session().run([a,b])

print result
