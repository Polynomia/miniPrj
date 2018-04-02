import resnet20,getData,os.path
import tensorflow as tf 
from datetime import datetime

with tf.device('/gpu:0'):
    flags = tf.flags
    FLAGS = flags.FLAGS
 #   flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for the training.')
    flags.DEFINE_integer('max_epoches', 200, 'Number of epoches to run trainer.')
    flags.DEFINE_integer('batch_size', 128,
    'Batch size. Must divide dataset sizes without remainder.')
    flags.DEFINE_string('train_dir', 'res_logs',
    'Directory to put the training data.')

    learning_rate = 0.001
    # Put logs for each run in separate directory
    train_logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/train/'
    test_logdir = FLAGS.train_dir + '/'  + datetime.now().strftime('%Y%m%d-%H%M%S') + '/test/'

    # Define input placeholders
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3),name='images')
    labels_placeholder = tf.placeholder(tf.int64, shape=None, name='image-labels')
    keeprob_placeholder = tf.placeholder(tf.float32, shape=None, name='keep_prob')
    isTrain_placeholder = tf.placeholder(tf.bool, name='phase_train')

    # Operation for the classifier's result
    logits = resnet20.residual_net(images_placeholder, keeprob_placeholder,isTrain_placeholder)

    # Operation for the loss function
    loss = resnet20.loss(logits, labels_placeholder)

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Operation for the training step
    train_step = resnet20.training(loss, learning_rate, global_step)

    # Operation calculating the accuracy of our predictions
    accuracy = resnet20.evaluation(logits, labels_placeholder)

    # Operation merging summary data for TensorBoard
    merged = tf.summary.merge_all()

    # Define saver to save model state at checkpoints
    saver = tf.train.Saver()

    # Load CIFAR-10 data
    data_sets = getData.load_cifar10()


    # -----------------------------------------------------------------------------
    # Run the TensorFlow graph
    # -----------------------------------------------------------------------------

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)

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
                    labels_placeholder: labels_batch,
                    keeprob_placeholder: 0.5,
                    isTrain_placeholder: True
                }
                # train_loss = sess.run(loss, feed_dict=feed_dict)
                # print('Step {:d}, loss {:g}'.format(j, train_loss))
                # Perform a single training step
                sess.run([train_step, loss], feed_dict=feed_dict)
                

                # Periodically save checkpoint

            # Periodically print out the model's current accuracy
            summary, train_accuracy , train_loss= sess.run([merged,accuracy,loss], feed_dict=feed_dict)
            print('Epoch {:d}, training accuracy {:g}'.format(i, train_accuracy))
            print('Epoch {:d}, training loss {:g}'.format(i, train_loss))
            train_writer.add_summary(summary, i)
            summary, test_accuracy, test_loss = sess.run([merged, accuracy, loss], feed_dict={images_placeholder: data_sets['test_data'],
                                                        labels_placeholder: data_sets['test_label'], keeprob_placeholder: 1,isTrain_placeholder: False})
            test_writer.add_summary(summary, i)
            print('Test accuracy {:g}'.format(test_accuracy))
            print('Test loss {:g}'.format(test_loss))

            # if i == 1:
            #     learning_rate *= 0.1
                # train_step = resnet20.training(loss, learning_rate, global_step)
            if (i+1)%10 == 0:
                learning_rate *= 0.1
                train_step = resnet20.training(loss, learning_rate, global_step)
            # if (i + 1) % 20 == 0:
            #     checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
            #     saver.save(sess, checkpoint_file, global_step=i)
            #     print('Saved checkpoint') 
            #     learning_rate = learning_rate*0.8
            #     train_step = resnet20.training(loss, learning_rate, global_step)
            
