import tensorflow as tf

participate = ["172.20.1.34:2428", "172.20.1.35:2428"]

cluster = tf.train.ClusterSpec({"pt":participate})

tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

server = tf.train.Server(cluster, job_name="pt", task_index=FLAGS.task_index)

if FLAGS.task_index == 0 :
    with tf.device("/job:pt/task:0") :
        a = tf.constant([1, 2, 3])

    with tf.device("/job:pt/task:1") :
        b = tf.constant([10, 20, 30])
        add = a + b
    sess = tf.Session("grpc://172.20.1.34:2428", config=tf.ConfigProto(log_device_placement=True))
    print sess.run(add)

else :
    server.join()
    
