import numpy as np
import pickle
import sys

def unpickle_cifar10_batch(file):
    '''Unpicke the cifar10 data from a single batch file. 
    Args:
        file: path to the file.
    Returns: 
        x,y: the np array of image data and the np array of labels
    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict['data'].astype(float), np.array(dict['labels'])

def load_cifar10():
    '''load all the cifar10 data. 
    Returns:
        data: {
                "train_data": x_train,
                "train_label": y_train,
                "test_data": x_test,
                "test_label": y_test,
                "classes": number of classes(10)
            }
    '''
    xs = []
    ys = []
    for i in range(1,6):
        file = 'cifar-10-batches-py/data_batch_{}'.format(i)
        x,y  = unpickle_cifar10_batch(file)
        xs.append(x)
        ys.append(y)
    
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    x_test, y_test = unpickle_cifar10_batch('cifar-10-batches-py/test_batch')

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']

    # Normalize Data
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image

    # Reshape the data
    x_train = np.transpose(x_train.reshape((-1, 3, 32, 32)), (0,2,3,1))
    x_test = np.transpose(x_test.reshape((-1, 3, 32, 32)), (0,2,3,1))

    data = {
        "train_data": x_train,
        "train_label": y_train,
        "test_data": x_test,
        "test_label": y_test,
        "classes": classes
    }
    
    return data  

def load_more_data():
    '''load all the cifar10 data. 
    Returns:
        data: {
                "train_data": x_train,
                "train_label": y_train,
                "test_data": x_test,
                "test_label": y_test,
                "classes": number of classes(10)
            }
    '''
    xs = []
    ys = []
    for i in range(1,6):
        file = 'cifar-10-batches-py/data_batch_{}'.format(i)
        x,y  = unpickle_cifar10_batch(file)
        xs.append(x)
        ys.append(y)
    
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    x_test, y_test = unpickle_cifar10_batch('cifar-10-batches-py/test_batch')

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']

    # Normalize Data
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image

    # Reshape the data
    x_train = np.transpose(x_train.reshape((-1, 3, 32, 32)), (0,2,3,1))
    x_test = np.transpose(x_test.reshape((-1, 3, 32, 32)), (0,2,3,1))

    x_train = np.concatenate((x_train[:,:28,:28,:],x_train[:,4:,4:,:]),axis=0)
    y_train = np.concatenate((y_train,y_train),axis=0)

    x_test = x_test[:,:28,:28,:]

    

    data = {
        "train_data": x_train,
        "train_label": y_train,
        "test_data": x_test,
        "test_label": y_test,
        "classes": classes
    }
    
    return data


def gen_batch(data, batch_size, epoches):
    '''Get data in batches. 
    Args:
        data: the zip of data and labels
        batch_size: size of batch
        num_iter: number of iters you want
    Returns:
        a generator of batch-data
    '''

    data = np.array(data)
    index = len(data)
    while True:
        index += batch_size
        if (index + batch_size > len(data)):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffled_indices]
        yield data[index:index + batch_size]  


if __name__ == '__main__':
    data = load_more_data()
    i = 1
    print(data['train_data'].shape)
    print(data['test_data'].shape)