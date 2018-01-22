'''
Distributed Tensorflow 0.8.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time
from pdb import set_trace
from tensorflow.contrib.training.python.training import device_setter as device_setter_lib


batch_size = 100
learning_rate = 0.0005
training_epochs = 20
logs_path = "/home/jwl1993/log/mnist/"

tf.app.flags.DEFINE_string('job_name', 'ps', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                            """parameter server jobs,"""
                            """'machine1:2222,machine2:2222,machine3:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                            """worker server jobs,"""
                            """'machine1:2222,machine2:2222,machine3:2222'""")
tf.app.flags.DEFINE_integer('task_index',0,'index of task')

FLAGS = tf.app.flags.FLAGS




#ps = ["172.20.1.35:2428"]
ps = FLAGS.ps_hosts.split(",")
#worker = ["172.20.1.36:2428", "172.20.1.37:2428"]      #dumbo35, dumbo36
worker = FLAGS.worker_hosts.split(",")
# load mnist data set
cluster = tf.train.ClusterSpec({"ps":ps, "worker":worker})


server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)

        # input images
        with tf.name_scope('input'):
          x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
          y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))

        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x,W1),b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2,W2),b2)
            y  = tf.nn.softmax(z3)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        # specify optimizer
        with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)
            
        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session 
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        print("task number is %i" % FLAGS.task_index)
        print("Variables initialized ...")

        begin_time = time.time()
        frequency = 100

        sess = tf.Session("grpc://172.20.1.34:2428", config=tf.ConfigProto(log_device_placement=True))
        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        start_time = time.time()
        for epoch in range(training_epochs):
            batch_count = int(mnist.train.num_examples/batch_size)
            count = 0
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                _, cost, summary, step = sess.run(
                                                [train_op, cross_entropy, summary_op, global_step], 
                                                feed_dict={x: batch_x, y_: batch_y} )
                writer.add_summary(summary, step)
                count += 1
                if count % frequency == 0 or i+1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step+1), 
                                " Epoch: %2d," % (epoch+1), 
                                " Batch: %3d of %3d," % (i+1, batch_count), 
                                " Cost: %.4f," % cost, 
                                " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                    count = 0
            print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            print("Final Cost: %.4f" % cost)
        server.join()

print("done")


