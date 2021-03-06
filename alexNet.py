import tensorflow as tf
import numpy as np 
import getData
_NUM_CLASS = 10

def lrn(name, previous_layer):
    '''Sets up the Local Response Normalization. 
    Args:
        name: namescope of this layer.
        previous_layer: The input layer of Local Response Normalization.
    Returns:
        lrn: The lrn layer.
    '''
    with tf.name_scope(name) as scope:
        lrn = tf.nn.local_response_normalization(previous_layer,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)
        tf.summary.histogram('lrn', lrn)
    return lrn
 
def conv(name, previous_layer, config):
    '''Sets up the convolutional layer.
    Args:
        name: namescope of this layer.
        previous_layer: The input layer of convolutional layer.
        config: {
                    'filter_height' : the height of filter,
                    'filter_width' : the width of filter,
                    'stride_height' : the height of stride,
                    'stride_width' : the width of stride,
                    'in_channels' : the channel of input data,
                    'out_channels' : number of filters,
                    'bias' : the value of bias(0 or 1 in AlexNet)
                }
    Returns:
        conv_relu: The conv layer after relu.
    '''
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal((config['filter_height'],  config['filter_width'], 
                                                config['in_channels'], config['out_channels']),                                               
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(previous_layer, kernel, [1, config['stride_height'], config['stride_width'], 1], padding='SAME')
        # bias = tf.Variable(tf.constant(config['bias'], shape=[config['out_channels']], dtype=tf.float32),
                            # trainable=True, name='bias')
        bias = tf.Variable(tf.random_normal([config['out_channels']], stddev=0.35), trainable=True, name='bias')
        conv_relu = tf.nn.relu(bias+conv, name = name)
        tf.summary.histogram('conv_relu',conv_relu)
        tf.summary.histogram('conv', conv)
    return conv_relu

def pool(name, previous_layer, config):
    '''Sets up the max-pooling layer.
    Args:
        name: namescope of this layer.
        previous_layer: The input layer of max-pooling layer.
        config: {
                    'filter_height' : the height of filter,
                    'filter_width' : the width of filter,
                    'stride_height' : the height of stride,
                    'stride_width' : the width of stride,
                }
    Returns:
        pool: The max-pooling layer.
    '''
    with tf.name_scope(name) as scope:
        pool = tf.nn.max_pool(previous_layer,
                            ksize=[1, config['filter_height'], config['filter_width'], 1],
                            strides=[1, config['stride_height'], config['stride_width'], 1],
                            padding='SAME')   
        tf.summary.histogram('pool', pool)
    return pool   

def fc(name, previous_layer, config):
    '''Sets up the fully connected layer.
    Args:
        name: namescope of this layer.
        previous_layer: The input layer of fully connected layer.
        config: {
                    'input' : size of input,
                    'output' : size of output,
                }
    Returns:
        fc_relu: The fully connected layer after relu.
    '''
    with tf.name_scope(name) as scope:
        weights = tf.Variable(tf.random_normal(shape=(config['input'], config['output']), 
                                                mean=0, stddev=1.0), trainable=True, name='weights')
        # bias = tf.Variable(tf.constant(1, shape=[config['output']], dtype=tf.float32),
                            # trainable=True, name='bias')
        bias = tf.Variable(tf.random_normal([config['output']], stddev=0.35), trainable=True, name='bias')
        fc_relu = tf.nn.relu(tf.matmul(previous_layer, weights)+bias)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('fc_relu', fc_relu)
    return fc_relu 


def inference(images):
    '''Build the model up to where it may be used for inference.
    Args:
        images: Images placeholder (input data).
    Returns:
        logits: Output tensor containing the computed logits.
    '''

    # conv1
    # config_conv1 = {
    #     'filter_height' : 11,
    #     'filter_width' : 11,
    #     'stride_height' : 4,
    #     'stride_width' : 4,
    #     'in_channels' : 3,
    #     'out_channels' : 64,
    #     'bias' : 0
    # }
    # conv1 = conv('conv1', images, config_conv1)
    conv1 = tf.layers.conv2d(
        inputs=images, 
        filters=64, 
        kernel_size=[11, 11],
        strides=[4, 4],
        padding='same', 
        activation=tf.nn.relu)
    tf.summary.histogram('conv1', conv1)

    # lrn1 = lrn('lrn1', conv1)
    lrn1 = tf.nn.local_response_normalization(
        conv1,
        alpha=1e-4,
        beta=0.75,
        depth_radius=2,
        bias=2.0)
    tf.summary.histogram('lrn1', lrn1)

    # config_pool1 = {
    #     'filter_height' : 3,
    #     'filter_width' : 3,
    #     'stride_height' : 2,
    #     'stride_width' : 2,
    # }
    # pool1 = pool('pool1', lrn1, config_pool1)
    pool1 = tf.layers.max_pooling2d(
        inputs=lrn1,
        pool_size=[3, 3],
        strides=2,
        padding='same')
    tf.summary.histogram('pool1', pool1)

    # config_conv2= {
    #     'filter_height' : 5,
    #     'filter_width' : 5,
    #     'stride_height' : 1,
    #     'stride_width' : 1,
    #     'in_channels' : 64,
    #     'out_channels' : 192,
    #     'bias' : 0
    # }
    # conv2 = conv('conv2', pool1, config_conv2)
    conv2 = tf.layers.conv2d(
        inputs=pool1, 
        filters=192, 
        kernel_size=[5, 5],
        padding='same', 
        activation=tf.nn.relu)
    tf.summary.histogram('conv2', conv2)

    # lrn2 = lrn('lrn2', conv2)
    lrn2 = tf.nn.local_response_normalization(
        conv2,
        alpha=1e-4,
        beta=0.75,
        depth_radius=2,
        bias=2.0)
    tf.summary.histogram('lrn2', lrn2)
    
    # config_pool2 = {
    #     'filter_height' : 3,
    #     'filter_width' : 3,
    #     'stride_height' : 2,
    #     'stride_width' : 2,
    # }
    # pool2 = pool('pool2', lrn2, config_pool2)
    pool2 = tf.layers.max_pooling2d(
        inputs=lrn2,
        pool_size=[3, 3],
        strides=2,
        padding='same')
    tf.summary.histogram('pool2', pool2)

    # config_conv3 = {
    #     'filter_height' : 3,
    #     'filter_width' : 3,
    #     'stride_height' : 1,
    #     'stride_width' : 1,
    #     'in_channels' : 192,
    #     'out_channels' : 384,
    #     'bias' : 0
    # }
    # conv3 = conv('conv3', pool2, config_conv3) 
    conv3 = tf.layers.conv2d(
        inputs=pool2, 
        filters=384, 
        kernel_size=[3, 3],
        padding='same', 
        activation=tf.nn.relu)
    tf.summary.histogram('conv3', conv3)

    # config_conv4 = {
    #     'filter_height' : 3,
    #     'filter_width' : 3,
    #     'stride_height' : 1,
    #     'stride_width' : 1,
    #     'in_channels' : 384,
    #     'out_channels' : 256,
    #     'bias' : 0
    # }
    # conv4 = conv('conv4', conv3, config_conv4) 
    conv4 = tf.layers.conv2d(
        inputs=conv3, 
        filters=256, 
        kernel_size=[3, 3],
        padding='same', 
        activation=tf.nn.relu)
    tf.summary.histogram('conv4', conv4)

    # config_conv5 = {
    #     'filter_height' : 3,
    #     'filter_width' : 3,
    #     'stride_height' : 1,
    #     'stride_width' : 1,
    #     'in_channels' : 256,
    #     'out_channels' : 256,
    #     'bias' : 0
    # }
    # conv5 = conv('conv5', conv4, config_conv5) 
    conv5 = tf.layers.conv2d(
        inputs=conv4, 
        filters=256, 
        kernel_size=[3, 3],
        padding='same', 
        activation=tf.nn.relu)
    tf.summary.histogram('conv5', conv5)

    # config_pool5 = {
    #     'filter_height' : 3,
    #     'filter_width' : 3,
    #     'stride_height' : 2,
    #     'stride_width' : 2,
    # }
    # pool5 = pool('pool5', conv5, config_pool5)
    pool5 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size=[3, 3],
        strides=2,
        padding='same')
    tf.summary.histogram('pool5', pool5)


    # fc layers
    reshape = tf.reshape(pool5, [-1, 256])

    # config = {
    #     'input' : 256,
    #     'output' : 384
    # }  
    # fc1 = fc('fc1', reshape, config)
    fc1 = tf.layers.dense(
        inputs=reshape, 
        units=384, 
        activation=tf.nn.relu)
    tf.summary.histogram('fc1', fc1)

    # config = {
    #     'input' : 384,
    #     'output' : _NUM_CLASS
    # }  
    # fc2 = fc('fc2', fc1, config)
    # print(tf.shape(fc2))
    fc2 = tf.layers.dense(
        inputs=fc1, 
        units=_NUM_CLASS, 
        activation=tf.nn.relu)
    tf.summary.histogram('fc2', fc2)

    return fc2

def loss(logits, labels):
    '''Calculates the loss from logits and labels.
    Args:
        logits: Logits tensor, float - [batch size, number of classes].
        labels: Labels tensor, int64 - [batch size].
    Returns:
        loss: Loss tensor of type float.
    '''

    with tf.name_scope('Loss'):
        # print(tf.shape(logits))
        # print(tf.shape(labels))
        # Operation to determine the cross entropy between logits and labels
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=logits, labels=labels, name='cross_entropy'))

        # Add a scalar summary for the loss
        tf.summary.scalar('loss', loss)

    return loss


def training(loss, learning_rate, global_step):
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
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    return train_step

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
    data_sets = getData.load_cifar10()

    # Define input placeholders
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3),name='images')
    labels_placeholder = tf.placeholder(tf.int64, shape=None, name='image-labels')
    # Operation for the classifier's result
    logits = inference(images_placeholder)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Generate input data batches
        zipped_data = zip(data_sets['train_data'], data_sets['train_label'])
        batches = getData.gen_batch(list(zipped_data), 128, 20)
        batch = next(batches)
        images_batch, labels_batch = zip(*batch)
        feed_dict = {
            images_placeholder: np.array(images_batch),
            labels_placeholder: np.array(labels_batch)
        }
        sess.run(logits, feed_dict)
