import old_alexNet,getData,os.path
import tensorflow as tf 
from datetime import datetime
import time,resource,sys
import numpy as np

with tf.device('/gpu:3'):
    flags = tf.flags
    FLAGS = flags.FLAGS
 #   flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for the training.')
    flags.DEFINE_integer('max_epoches', 200, 'Number of epoches to run trainer.')
    flags.DEFINE_integer('batch_size', 60,
    'Batch size. Must divide dataset sizes without remainder.')
    flags.DEFINE_string('train_dir', 'tf_logs',
    'Directory to put the training data.')

    learning_rate = 0.01

    # Put logs for each run in separate directory
    train_logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/train/'
    test_logdir = FLAGS.train_dir + '/'  + datetime.now().strftime('%Y%m%d-%H%M%S') + '/test/'

    # Define input placeholders
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3),name='images')
    labels_placeholder = tf.placeholder(tf.int64, shape=None, name='image-labels')
    keeprob_placeholder = tf.placeholder(tf.float32, shape=None, name='keep_prob')
    isTrain_placeholder = tf.placeholder(tf.bool, name='phase_train')
    # Operation for the classifier's result
    logits = old_alexNet.inference(images_placeholder, keeprob_placeholder,isTrain_placeholder)
    print("[logits] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
    # Operation for the loss function
    loss = old_alexNet.loss(logits, labels_placeholder)
    print("[loss] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Operation for the training step
    train_step = old_alexNet.training(loss, learning_rate, global_step)

    # Operation calculating the accuracy of our predictions
    accuracy = old_alexNet.evaluation(logits, labels_placeholder)
    print("[acc] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
    # Operation merging summary data for TensorBoard
    merged = tf.summary.merge_all()
    print("[merge] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
    # Define saver to save model state at checkpoints
    saver = tf.train.Saver()
    print("[saver] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
    # Load CIFAR-10 data
    data_sets = getData.load_cifar10()
    print("[load data] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
    # Generate input data batches
    zipped_data = zip(data_sets['train_data'], data_sets['train_label'])
    batches = getData.gen_batch(list(zipped_data), FLAGS.batch_size,FLAGS.max_epoches)
    print("[genBatch] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
    # -----------------------------------------------------------------------------
    # Run the TensorFlow graph
    # -----------------------------------------------------------------------------
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)

        
        for i in range(FLAGS.max_epoches):
            for j in range(data_sets['train_data'].shape[0]//FLAGS.batch_size):
                # Get next input data batch
                batch = next(batches)
                images_batch, labels_batch = zip(*batch)
                feed_dict = {
                    images_placeholder: images_batch,
                    labels_placeholder: labels_batch,
                    keeprob_placeholder: 0.5,
                    isTrain_placeholder: True
                }
                # train_loss = sess.run(loss, feed_dict=feed_dict)
                # print('Step {:d}, loss {:g}'.format(j, train_loss))
                # Perform a single training step
                _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)
                

                # Periodically save checkpoint

            # Periodically print out the model's current accuracy
            print("[train] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
            summary, train_accuracy = sess.run([merged,accuracy], feed_dict=feed_dict)
            print("[epochtrain] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
            print('Epoch {:d}, training accuracy {:g}'.format(i, train_accuracy))
            train_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=train_accuracy), tf.Summary.Value(tag="loss", simple_value=train_loss)])
            # train_summary.Value.add(tag='accuracy', simple_value=train_accuracy)
            # train_summary.Value.add(tag='loss', simple_value=train_loss)
            train_writer.add_summary(summary, i)
            train_writer.add_summary(train_summary,i)
            acc = []
            test_losses = []
            for k in range(data_sets['test_data'].shape[0]//FLAGS.batch_size):
                test_feed_dic = {
                    images_placeholder: data_sets['test_data'][i*FLAGS.batch_size:(i+1)*FLAGS.batch_size],
                    labels_placeholder: data_sets['test_label'][i*FLAGS.batch_size:(i+1)*FLAGS.batch_size],
                    keeprob_placeholder: 0.5,
                    isTrain_placeholder: False
                }
                test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict=test_feed_dic)
                acc.append(test_accuracy)
                test_losses.append(test_loss)
            avg_acc  = float(np.mean(np.asarray(acc)))
            avg_loss = float(np.mean(np.asarray(test_losses)))
            #test_summary = tf.Summary()
            test_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=avg_acc), tf.Summary.Value(tag="loss", simple_value=avg_loss)])
            # test_summary.Value.add(tag='accuracy', simple_value=avg_acc)
            # test_summary.Value.add(tag='loss', simple_value=avg_loss)
            test_writer.add_summary(test_summary, i)
            print("[epochtest] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)
            print('Test accuracy {:g}'.format(test_accuracy))
               
            if (i + 1) % 20 == 0:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=i)
                print('Saved checkpoint') 
                learning_rate = learning_rate*0.8
                train_step = old_alexNet.training(loss, learning_rate, global_step)
            

