import os
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse
from utils import util


def load_multiview_data(config):
    data_name = config['dataset']
    main_dir = sys.path[0]
    x = None
    y = None

    if data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse_21.mat'))
        x = mat['X'][0]
        x[0] = sparse.csr_matrix(mat['X'][0, 0]).A  # 20
        x[1] = sparse.csr_matrix(mat['X'][0, 1]).A  # 59
        x[2] = sparse.csr_matrix(mat['X'][0, 2]).A  # 40
        y = np.squeeze(mat['Y']).astype('int') - 1

    elif data_name in ['Scene_15', 'Caltech101-20', 'Caltech101-7', 'MSRC_v1', 'NUS', 'NoisyMNIST_select']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        x = mat['X'][0]
        num_views = len(x)
        for view in range(num_views):
            x[view] = util.normalize(x[view]).astype('float32')
        y = np.squeeze(mat['Y']).astype('int')
        y = y - np.min(y)

    elif data_name in ['NoisyMNIST']:
        data = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        # train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
        x = []
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
        x.append(np.concatenate([tune.images1, test.images1], axis=0))
        x.append(np.concatenate([tune.images2, test.images2], axis=0))
        y = np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]) - 1

    elif data_name in ['20newsgroups']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        x = mat['data'][0]
        num_views = len(x)
        for view in range(num_views):
            x[view] = util.normalize(x[view].T).astype('float32')
        y = np.squeeze(mat['truelabel'][0][0].T).astype('int') - 1

    return x, y


class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # assert images.shape[3] == 1
            # images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels
