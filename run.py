import torch
import itertools
import argparse
import time
import os
import numpy as np
from utils.util import cal_classify
from utils.logger_ import get_logger
from utils.datasets import load_multiview_data
import collections
import warnings
import random
from CIML import CIML
from sklearn.model_selection import train_test_split
from configure.configure_supervised_multiview import get_default_config

warnings.simplefilter("ignore")

dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "LandUse_21",
    3: "NoisyMNIST_select",
    4: "NUS",
    5: "MSRC_v1"
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='10', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)  # 使用第一, 三块GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger, plt_name = get_logger(config)
    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X, Y = load_multiview_data(config)

    # split data
    INDEX = list(range(X[0].shape[0]))
    INDEX_train, INDEX_test = train_test_split(INDEX, test_size=0.2, random_state=42)

    fold_acc, fold_precision, fold_f_measure = [], [], []
    for data_seed in range(1, args.test_time + 1):
        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        # Set random seeds
        seed = data_seed - 1 + config['seed']
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # 运行10次
        start = time.time()
        # Build model
        model_CIML = CIML(config)
        optimizer = torch.optim.Adam(
            itertools.chain(model_CIML.autoencoders.parameters(), model_CIML.f.parameters(),
                            model_CIML.fc.parameters(), model_CIML.classifier.parameters(), [model_CIML.c]),
            lr=config['training']['lr'])

        logger.info(model_CIML.autoencoders[0])
        logger.info(model_CIML.f[0])
        logger.info(optimizer)

        model_CIML.autoencoders.to(device), model_CIML.f.to(device), model_CIML.fc.to(device), model_CIML.c.to(device)
        model_CIML.classifier.to(device)

        acc, precision, f_measure = model_CIML.train_con_spe_accelerate(config, logger, accumulated_metrics, X, Y,
                                                             INDEX_train, INDEX_test, device, optimizer)

        fold_acc.append(acc)
        fold_precision.append(precision)
        fold_f_measure.append(f_measure)
        print(time.time() - start)
        print('run ', data_seed, 'times!')

    logger.info('--------------------Training over--------------------')
    acc, precision, f_measure = cal_classify(logger, fold_acc, fold_precision, fold_f_measure)


if __name__ == '__main__':
    main()
