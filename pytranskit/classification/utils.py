import numpy as np
from scipy.io import loadmat
import os
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split

def new_index_matrix(max_index, n_samples_perclass, num_classes, repeat, y_train):
    seed = int('{}{}{}'.format(n_samples_perclass, num_classes, repeat))
    np.random.seed(seed)
    index = np.zeros([num_classes, n_samples_perclass], dtype=np.int64)
    for classidx in range(num_classes):
        max_samples = (y_train == classidx).sum()
        index[classidx] = np.random.randint(0, max_samples, (n_samples_perclass))
    return index


def take_samples(data, labels, index, num_classes):
    assert data.shape[0] == labels.shape[0]
    assert index.shape[0] == num_classes
    indexed_data = []
    new_labels = []
    for i in range(num_classes):
       class_data, class_labels = data[labels == i], labels[labels == i]
       indexed_data.append(class_data[index[i]])
       new_labels.append(class_labels[index[i]])
    return np.concatenate(indexed_data), np.concatenate(new_labels)


def load_data(dataset, num_classes, datadir='data'):
    cache_file = os.path.join(datadir, dataset, 'dataset.hdf5')
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            x_train, y_train = f['x_train'][()], f['y_train'][()]
            x_test, y_test = f['x_test'][()], f['y_test'][()]
            print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
            return (x_train, y_train), (x_test, y_test)

    print('loading data from mat files')
    x_train, y_train, x_test, y_test = [], [], [], []
    for split in ['training', 'testing']:
        for classidx in range(num_classes):
            datafile = os.path.join(datadir, dataset, '{}/dataORG_{}.mat'.format(split, classidx))
            # loadmat(datafile)['xxO'] is of shape (H, W, N)
            data = loadmat(datafile)['xxO'].transpose([2, 0, 1]) # transpose to (N, H, W)
            label = np.zeros(data.shape[0], dtype=np.int64)+classidx
            #print('split {} class {} data.shape {}'.format(split, classidx, data.shape))
            if split == 'training':
                x_train.append(data)
                y_train.append(label)
            else:
                x_test.append(data)
                y_test.append(label)
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
    print('x_train.shape {} x_test.shape {}'.format(x_train.shape, x_test.shape))

    x_train = x_train / x_train.max(axis=(1, 2), keepdims=True)
    x_test = x_test / x_test.max(axis=(1, 2), keepdims=True)

    x_train = (x_train * 255.).astype(np.uint8)
    x_test = (x_test * 255.).astype(np.uint8)

    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('x_train', data=x_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('x_test', data=x_test)
        f.create_dataset('y_test', data=y_test)
        print('saved to {}'.format(cache_file))

    return (x_train, y_train), (x_test, y_test)

def load_data_1D(dataset, num_classes, datadir='data'):
    cache_file = os.path.join(datadir, dataset, 'dataset.hdf5')
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            x_train, y_train = f['x_train'][()], f['y_train'][()]
            x_test, y_test = f['x_test'][()], f['y_test'][()]
            print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
            return (x_train, y_train), (x_test, y_test)

    print('loading data from mat files')
    x_train, y_train, x_test, y_test = [], [], [], []
    for split in ['training', 'testing']:
        for classidx in range(num_classes):
            datafile = os.path.join(datadir, dataset, '{}/dataORG_{}.mat'.format(split, classidx))
            # loadmat(datafile)['xxO'] is of shape (H, W, N)
            data = loadmat(datafile)['xxO'].T
            label = np.zeros(data.shape[0], dtype=np.int64)+classidx
            #print('split {} class {} data.shape {}'.format(split, classidx, data.shape))
            if split == 'training':
                x_train.append(data)
                y_train.append(label)
            else:
                x_test.append(data)
                y_test.append(label)
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
    print('x_train.shape {} x_test.shape {}'.format(x_train.shape, x_test.shape))

    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('x_train', data=x_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('x_test', data=x_test)
        f.create_dataset('y_test', data=y_test)
        print('saved to {}'.format(cache_file))

    return (x_train, y_train), (x_test, y_test)

def load_data_3D(dataset, num_classes, datadir='data'):
    cache_file = os.path.join(datadir, dataset, 'dataset.hdf5')
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            x_train, y_train = f['x_train'][()], f['y_train'][()]
            x_test, y_test = f['x_test'][()], f['y_test'][()]
            print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
            return (x_train, y_train), (x_test, y_test)

    print('loading data from mat files')
    x_train, y_train, x_test, y_test = [], [], [], []
    for split in ['training', 'testing']:
        for classidx in range(num_classes):
            datafile = os.path.join(datadir, dataset, '{}/dataORG_{}.mat'.format(split, classidx))
            # loadmat(datafile)['xxO'] is of shape (H, W, N)
            data = loadmat(datafile)['xxO'].transpose([3, 0, 1, 2]) # transpose to (N, H, W, D)
            label = np.zeros(data.shape[0], dtype=np.int64)+classidx
            print('split {} class {} data.shape {}'.format(split, classidx, data.shape))
            if split == 'training':
                x_train.append(data)
                y_train.append(label)
            else:
                x_test.append(data)
                y_test.append(label)
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
    print('x_train.shape {} x_test.shape {}'.format(x_train.shape, x_test.shape))

    x_train = x_train / x_train.max(axis=(1, 2, 3), keepdims=True)
    x_test = x_test / x_test.max(axis=(1, 2, 3), keepdims=True)

    x_train = (x_train * 255.).astype(np.float32)
    x_test = (x_test * 255.).astype(np.float32)

    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('x_train', data=x_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('x_test', data=x_test)
        f.create_dataset('y_test', data=y_test)
        print('saved to {}'.format(cache_file))

    return (x_train, y_train), (x_test, y_test)


def take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat):
    max_index = x_train.shape[0] // num_classes
    train_index = new_index_matrix(max_index, n_samples_perclass, num_classes, repeat, y_train)
    x_train_sub, y_train_sub = take_samples(x_train, y_train, train_index, num_classes)
    return x_train_sub, y_train_sub

def take_train_val_samples(x_train, y_train, n_samples_perclass, num_classes, repeat):
    max_index = x_train.shape[0]//num_classes
    train_index = new_index_matrix(max_index, n_samples_perclass, num_classes, repeat, y_train)

    val_samples = n_samples_perclass // 10 # Use 10% for validation

    if val_samples >= 1:
        val_index = train_index[:, :val_samples]
        x_val, y_val = take_samples(x_train, y_train, val_index, num_classes)
        assert x_val.shape[0] == y_val.shape[0]
        print('validation data shape {}'.format(x_val.shape), end=' ')
    else:
        x_val, y_val = None, None
        print('validation data {}'.format(x_val), end=' ')

    train_sub_index = train_index[:, val_samples:]
    x_train_sub, y_train_sub = take_samples(x_train, y_train, train_sub_index, num_classes)
    print('train data shape {}'.format(x_train_sub.shape))

    if x_val is not None:
        assert x_val.shape[0] + x_train_sub.shape[0] == n_samples_perclass*num_classes
    else:
        assert x_train_sub.shape[0] == n_samples_perclass*num_classes


    return (x_train_sub, y_train_sub), (x_val, y_val)


def dataset_config(dataset):
    assert dataset in ['AffMNIST', 'LiverN', 'MNIST', 'OAM', 'OAM_t5', 'OAM_t10', 
                       'SignMNIST', 'Synthetic', 'CIFAR10', 'MNIST_outDist', 'HEP2']
    if dataset in ['MNIST']:
        rm_edge = True
        num_classes = 10
        po_train_max = 12  # maximum train samples = 2^po_max
        img_size = 28
    elif dataset in ['AffMNIST']:
        rm_edge = True
        num_classes = 10
        img_size = 84
        po_train_max = 12  # maximum train samples = 2^po_max
    elif dataset in ['OAM', 'OAM_t10', 'OAM_t5']:
        rm_edge = False
        num_classes = 32
        img_size = 151
        po_train_max = 9  # maximum train samples = 2^po_max
    elif dataset in ['SignMNIST']:
        rm_edge = False
        num_classes = 3
        img_size = 128
        po_train_max = 9  # maximum train samples = 2^po_max
    elif dataset in ['Synthetic']:
        rm_edge = True
        num_classes = 1000
        img_size = 64
        po_train_max = 4  # maximum train samples = 2^po_max
    elif dataset in ['LiverN']:
        rm_edge=False
        num_classes = 2
        img_size = 130
        po_train_max = 8   # maximum train samples = 2^po_max
    elif dataset in ['CIFAR10']:
        rm_edge = False
        num_classes = 10
        img_size = 32
        po_train_max = 12
    elif dataset in ['MNIST_outDist']:
        rm_edge = True
        num_classes = 10
        img_size = 84
        po_train_max = 12  # maximum train samples = 2^po_max
    elif dataset in ['HEP2']:
        rm_edge = False
        num_classes = 2
        img_size = 64
        po_train_max = 10  # maximum train samples = 2^po_max

    return num_classes, img_size, po_train_max, rm_edge
