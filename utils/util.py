import numpy as np
from torch.utils.data import Dataset


def normalize(x):
    """ Normalize """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def cal_classify(logger, *arg):
    """ print classification results """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     Precision {:.2f} std {:.2f} 
                     F-measure {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100,
                                                           np.std(arg[1]) * 100, np.mean(arg[2]) * 100,
                                                           np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)
    return


class matDataset(Dataset):
    def __init__(self, X, Y):
        self.data = {}
        for v, x in enumerate(X):
            self.data[v] = x
        self.Y = Y

    def __getitem__(self, index):
        data = []
        for x in self.data.values():
            data.append(x[index])
        y = self.Y[index]
        return data, y, index

    def __len__(self):
        return len(self.Y)
