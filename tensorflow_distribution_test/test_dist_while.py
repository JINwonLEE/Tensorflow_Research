import tensorflow as tf



participate = ["172.20.1.35:2428", "172.20.1.36:2428"]
cluster = tf.train.ClusterSpec({"pt" : participate})

tf.app.flags.DEFINE_integer("task", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

server = tf.train.Server(cluster, job_name="pt", task_index=FLAGS.task)

if FLAGS.task == 1 :
    print('[LJW] Task 1 (dumbo 36)')
    server.join()
elif FLAGS.task == 0 :
    print('[LJW] Task 0 (dumbo 35)')
    a, b = tf.while_loop(lambda a, b: a < 30,
        lambda a, b: (a * 3, b * 2),
        (2, 3))

    result = tf.Session("grpc://172.20.1.35:2428").run([a,b])

    print result
