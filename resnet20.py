import tensorflow as tf
import getData,math
def conv(x, n_in, n_out, kernel_size, stride, padding='SAME', bias=False, name='conv'):
    with tf.variable_scope(name) as scope:
        # kernel = tf.get_variable('weight', shape = [kernel_size,kernel_size,n_in,n_out], 
        #                         initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
        kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, n_in, n_out],stddev=math.sqrt(2/(kernel_size*kernel_size*n_in))),name='weight')
        tf.add_to_collection('weights', kernel)
        conv = tf.nn.conv2d(x, kernel, [1,stride,stride,1], padding=padding)
        if bias:
            bias = tf.get_variable('bias', [n_out], initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('biases', bias)
            conv = tf.nn.bias_add(conv, bias)
        tf.summary.histogram('conv',conv)
    return conv

def fc(previous_layer, n_in, n_out, name):
    with tf.variable_scope(name) as scope:
        # weights = tf.Variable(tf.random_normal(shape=(config['input'], config['output']), 
        #                                         mean=0, stddev=0.1), trainable=True, name='weights')
        # bias = tf.Variable(tf.constant(1.0, shape=[config['output']], dtype=tf.float32),
        #                     trainable=True, name='bias')
        weights = tf.get_variable('weights', shape=[n_in, n_out], initializer=tf.uniform_unit_scaling_initializer(factor=1.0),trainable=True)
        tf.add_to_collection('weights', weights)
        bias = tf.get_variable('bias', shape=[n_out],trainable=True)
        tf.add_to_collection('biases', bias)
        out = tf.nn.bias_add(tf.matmul(previous_layer, weights),bias)
        fc_relu = tf.nn.relu(out)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('fc_relu', fc_relu)
    return fc_relu 

def batch_norm(x, n_out, phase_train,name='batch_norm'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name) as scope:
    #     beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
    #                                  name='beta', trainable=True)
    #     gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
    #                                   name='gamma', trainable=True)
    #     batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    #     ema = tf.train.ExponentialMovingAverage(decay=0.9)

    #     def mean_var_with_update():
    #         ema_apply_op = ema.apply([batch_mean, batch_var])
    #         with tf.control_dependencies([ema_apply_op]):
    #             return tf.identity(batch_mean), tf.identity(batch_var)

    #     mean, var = tf.cond(phase_train,
    #                         mean_var_with_update,
    #                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
    #     normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    #     tf.summary.histogram('normed', normed)
    # return normed
        return tf.layers.batch_normalization(x,momentum=0.9,training=phase_train)

def residual_block(x, n_in, n_out, subsample, phase_train, scope='res_block'):
  with tf.variable_scope(scope):
    if subsample:
        y = conv(x, n_in, n_out, 3, 2, 'SAME', False, name='conv_1')
        shortcut = conv(x, n_in, n_out, 3, 2, 'SAME',
                    False, name='shortcut')
    else:
        y = conv(x, n_in, n_out, 3, 1, 'SAME', False, name='conv_1')
        shortcut = tf.identity(x, name='shortcut')
    y = batch_norm(y, n_out, phase_train, name='bn_1')
    y = tf.nn.relu(y, name='relu_1')
    y = conv(y, n_out, n_out, 3, 1, 'SAME', True, name='conv_2')
    y = batch_norm(y, n_out, phase_train, name='bn_2')
    y = y + shortcut
    y = tf.nn.relu(y, name='relu_2')
  return y

def residual_group(x, n_in, n_out, n, first_subsample, phase_train, scope='res_group'):
    with tf.variable_scope(scope):
        y = residual_block(x, n_in, n_out, first_subsample, phase_train, scope='block_1')
        for i in range(n - 1):
            y = residual_block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i + 2))
    return y

def residual_net(x, keep_prob, phase_train, scope='res_net'):
    n = 3
    n_classes = 10
    with tf.variable_scope(scope):
        y = conv(x, 3, 16, 3, 1, 'SAME', False, name='conv_init')
        y = batch_norm(y, 16, phase_train, name='bn_init')
        y = tf.nn.relu(y, name='relu_init')
        y = residual_group(y, 16, 16, n, False, phase_train, scope='group_1')
        y = residual_group(y, 16, 32, n, True, phase_train, scope='group_2')
        y = residual_group(y, 32, 64, n, True, phase_train, scope='group_3')
        y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
        print(y.get_shape())
        y = tf.reshape(y, [-1, 64])
        print(y.get_shape())
        y = fc(y,64,n_classes,'fc')
        print(y.get_shape())
    return y

def loss(logits, labels):
    '''Calculates the loss from logits and labels.
    Args:
        logits: Logits tensor, float - [batch size, number of classes].
        labels: Labels tensor, int64 - [batch size].
    Returns:
        loss: Loss tensor of type float.
    '''

    with tf.name_scope('Loss'):
        # Operation to determine the cross entropy between logits and labels
        vars   = tf.trainable_variables() 
        lossL2 = tf.add_n([tf.nn.l2_loss(o) for o in tf.get_collection('weights')])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=logits, labels=labels, name='cross_entropy')) + lossL2

        # Add a scalar summary for the loss
        tf.summary.scalar('loss', loss)

    return loss


def training(loss, learning_rate, my_global_step):
    '''Sets up the training operation.
    Creates an optimizer and applies the gradients to all trainable variables.
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_step: The op for training.
    '''


    # Create a gradient descent optimizer
    # (which also increments the global step counter)
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(
    #     loss, global_step=my_global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss, global_step=my_global_step)

    return train_step
    # params = tf.trainable_variables()
    # gradients = tf.gradients(loss, params, name='gradients')
    # optim = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # update = optim.apply_gradients(zip(gradients, params))
    # with tf.control_dependencies([update]):
    #     train_op = tf.no_op(name='train_op')
    # return train_op

def evaluation(logits, labels):
    '''Evaluates the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch size, number of classes].
        labels: Labels tensor, int64 - [batch size].
    Returns:
        accuracy: the percentage of images where the class was correctly predicted.
    '''

    with tf.name_scope('Accuracy'):
        # Operation comparing prediction with true label
        correct_prediction = tf.equal(tf.argmax(logits,1), labels)

        # Operation calculating the accuracy of the predictions
        accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summary operation for the accuracy
        tf.summary.scalar('accuracy', accuracy)

    return accuracy

if __name__ == '__main__':
    # Load CIFAR-10 data
    data_sets = getData.load_more_data()

    # Define input placeholders
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3),name='images')
    labels_placeholder = tf.placeholder(tf.int64, shape=None, name='image-labels')
    keeprob_placeholder = tf.placeholder(tf.float32, shape=None, name='keep_prob')
    isTrain_placeholder = tf.placeholder(tf.bool, name='phase_train')
    # Operation for the classifier's result
    logits = residual_net(images_placeholder, keeprob_placeholder, isTrain_placeholder)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Generate input data batches
        zipped_data = zip(data_sets['train_data'], data_sets['train_label'])
        batches = getData.gen_batch(list(zipped_data), 128, 20)
        batch = next(batches)
        images_batch, labels_batch = zip(*batch)

        print("ok")
