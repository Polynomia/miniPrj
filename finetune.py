import os,getData
import numpy as np
import tensorflow as tf
from vgg16 import vgg16
from datetime import datetime

with tf.device('/gpu:3'):
    flags = tf.flags
    FLAGS = flags.FLAGS
 #   flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for the training.')
    flags.DEFINE_integer('max_epoches', 10, 'Number of epoches to run trainer.')
    flags.DEFINE_integer('batch_size', 128,
    'Batch size. Must divide dataset sizes without remainder.')
    flags.DEFINE_string('train_dir', 'finetune_logs',
    'Directory to put the training data.')
    flags.DEFINE_string('weights', 'vgg16_weights.npz',
    'Pretrained model weights')

    learning_rate = 0.01

    # Put logs for each run in separate directory
    train_logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/train/'
    test_logdir = FLAGS.train_dir + '/'  + datetime.now().strftime('%Y%m%d-%H%M%S') + '/test/'

    # Define input placeholders
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3),name='images')
    labels_placeholder = tf.placeholder(tf.int64, shape=None, name='image-labels')

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    model = vgg16(images_placeholder,weights='vgg16_weights.npz')
    score = model.fc3

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score,labels=labels_placeholder))
        tf.summary.scalar('cross_entropy', loss)
    
    with tf.name_scope("train"):
        # Create optimizer and apply gradient descent to the trainable variables
        optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), labels_placeholder)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    
    merged = tf.summary.merge_all()

     # Define saver to save model state at checkpoints
    saver = tf.train.Saver()

    # Load CIFAR-10 data
    data_sets = getData.load_cifar10()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)
        model.load_weights(sess)
        # Generate input data batches
        zipped_data = zip(data_sets['train_data'], data_sets['train_label'])
        batches = getData.gen_batch(list(zipped_data), FLAGS.batch_size,FLAGS.max_epoches)
        for i in range(FLAGS.max_epoches):
            for j in range(data_sets['train_data'].shape[0]//FLAGS.batch_size):
                # Get next input data batch
                batch = next(batches)
                images_batch, labels_batch = zip(*batch)
                feed_dict = {
                    images_placeholder: images_batch,
                    labels_placeholder: labels_batch
                }
                # Perform a single training step
                sess.run([optimizer, loss], feed_dict=feed_dict)

                # Periodically save checkpoint

            # Periodically print out the model's current accuracy
            summary, train_accuracy = sess.run([merged,accuracy], feed_dict=feed_dict)
            print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
            train_writer.add_summary(summary, i)
            summary, test_accuracy = sess.run([merged, accuracy], feed_dict={images_placeholder: data_sets['test_data'],
                                                        labels_placeholder: data_sets['test_label']})
            test_writer.add_summary(summary, i)
            print('Test accuracy {:g}'.format(test_accuracy))
               
            if (i + 1) % 50 == 0:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=i)
                print('Saved checkpoint') 
                