import alexNet,getData,os.path
import tensorflow as tf 
from datetime import datetime

with tf.device('/cpu:0'):
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
    flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
    flags.DEFINE_integer('batch_size', 128,
    'Batch size. Must divide dataset sizes without remainder.')
    flags.DEFINE_string('train_dir', 'tf_logs',
    'Directory to put the training data.')

    # Put logs for each run in separate directory
    logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'


    # Define input placeholders
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3),name='images')
    labels_placeholder = tf.placeholder(tf.int64, shape=None, name='image-labels')

    # Operation for the classifier's result
    logits = alexNet.inference(images_placeholder)

    # Operation for the loss function
    loss = alexNet.loss(logits, labels_placeholder)

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Operation for the training step
    train_step = alexNet.training(loss, FLAGS.learning_rate, global_step)

    # Operation calculating the accuracy of our predictions
    accuracy = alexNet.evaluation(logits, labels_placeholder)

    # Operation merging summary data for TensorBoard
    summary = tf.summary.merge_all()

    # Define saver to save model state at checkpoints
    saver = tf.train.Saver()

    # Load CIFAR-10 data
    data_sets = getData.load_cifar10()


    # -----------------------------------------------------------------------------
    # Run the TensorFlow graph
    # -----------------------------------------------------------------------------

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        # Generate input data batches
        zipped_data = zip(data_sets['train_data'], data_sets['train_label'])
        batches = getData.gen_batch(list(zipped_data), FLAGS.batch_size,FLAGS.max_steps)
        
        for i in range(FLAGS.max_steps):

            # Get next input data batch
            batch = next(batches)
            images_batch, labels_batch = zip(*batch)
            feed_dict = {
                images_placeholder: images_batch,
                labels_placeholder: labels_batch
            }


            # Periodically print out the model's current accuracy
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)

            # Perform a single training step
            sess.run([train_step, loss], feed_dict=feed_dict)

            # Periodically save checkpoint
            if (i + 1) % 1000 == 0:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=i)
                print('Saved checkpoint')

            test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: data_sets['test_data'],
                                                            labels_placeholder: data_sets['test_label']})
            print('Test accuracy {:g}'.format(test_accuracy))

