import torch
import torch.autograd
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np
from model import Autoencoder, Classifier
from loss import cal_entropy, mut_info, kl_div, kl_norm, mut_info_y
from utils.util import matDataset


class CIML():
    def __init__(self, config):
        self._config = config
        self._latent_dim1 = config['Autoencoder1']['arch1'][0][-1]
        self._latent_dim2 = config['Autoencoder2']['arch1'][0][-1]
        self.view_num = config['view']
        self.autoencoders = torch.nn.ModuleList([Autoencoder(config['Autoencoder2']['arch1'][i],
                                                             config['Autoencoder2']['activation'],
                                                             config['Autoencoder2']['batchnorm'])
                                                 for i in range(self.view_num)])
        self.f = torch.nn.ModuleList([Autoencoder(config['Autoencoder1']['arch1'][i],
                                                  config['Autoencoder1']['activation'],
                                                  config['Autoencoder1']['batchnorm'])
                                      for i in range(self.view_num)])
        self.c = torch.normal(mean=torch.zeros([config['n'], self._latent_dim1]), std=1).to('cuda:0')
        self.c.requires_grad_(True)
        # save requires_grad state
        self.c_requires_grad_state = None
        self.fc = Autoencoder(config['Autoencoder1']['arch2'], config['Autoencoder1']['activation'],
                              config['Autoencoder1']['batchnorm'])
        self.classifier = Classifier(self._latent_dim2 * self.view_num + self._latent_dim1, config['class_num'])
        self.loss = torch.nn.CrossEntropyLoss()

    def train_con_spe(self, config, logger, accumulated_metrics, x, y, index_train, index_test, device, optimizer):
        x_train = []
        x_test = []

        for i in range(self.view_num):
            # x[i] = x[i].toarray()  # if run data Reuters, should add this code
            x_train_temp = torch.Tensor(x[i][index_train]).to(device)
            x_train.append(x_train_temp)
            x_test_temp = torch.Tensor(x[i][index_test]).to(device)
            x_test.append(x_test_temp)
        y_train = y[index_train]
        y_test = y[index_test]
        c_train = self.c[index_train]
        dataset = matDataset(x_train, y_train)
        train_loader = Data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
        epochs_total = config['training']['epoch']

        for epoch in range(epochs_total):
            # all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0
            for batch_idx, (batch_x, batch_y, idx) in enumerate(train_loader):
                # consistency
                for v in range(self.view_num):
                    batch_x[v] = batch_x[v].to(torch.float32).to(device)
                batch_c = c_train[idx].to(device)
                c_mu = batch_c.mean(dim=0)
                c_sigma = batch_c.std(dim=0)
                zc = self.fc.encoder(batch_c)
                zc_mu = zc.mean(dim=0)
                zc_sigma = zc.std(dim=0)
                I_zc_c = 0.5 * kl_div(c_mu, c_sigma, zc_mu, zc_sigma) + 0.5 * kl_div(zc_mu, zc_sigma, c_mu, c_sigma)
                # 将 batch_y 转换为 torch.float32 类型
                batch_y_float = batch_y.to(torch.float32).unsqueeze(dim=1).to(device)
                I_zc_y = mut_info(zc, batch_y_float, device)
                loss_mse = 0
                for v in range(self.view_num):
                    c_v = self.f[v].encoder(batch_x[v])
                    loss_mse += F.mse_loss(c_v, batch_c)
                loss_con = (- cal_entropy(c_sigma.to('cpu')) + loss_mse - I_zc_y + config['training']['lambda1'] * I_zc_c)
                # loss_con = (- reyi_entropy(x, sigma) + loss_mse - I_zc_y + config['training']['lambda1'] * I_zc_c)

                # print(loss_con)

                #  specificity
                zs = []
                I_zs_y = 0
                I_zs_x = 0
                I_zs_zc = 0
                I_zsi_zj = 0
                for v in range(self.view_num):
                    zs_v = self.autoencoders[v].encoder(batch_x[v])
                    zs.append(zs_v)
                    I_zs_y += mut_info(zs_v, batch_y_float, device)
                    zsv_mu = zs_v.mean()
                    zsv_sigma = zs_v.std()
                    I_zs_x += kl_norm(zsv_mu, zsv_sigma)
                    I_zs_zc += mut_info(zs_v, zc, device)

                for vi in range(self.view_num):
                    for vj in range(self.view_num):
                        if vi != vj:
                            I_zsi_zj += mut_info(zs[vi], zs[vj], device)

                loss_spe = - I_zs_y + config['training']['lambda2'] * I_zs_x + I_zs_zc + I_zsi_zj

                zs = torch.cat(zs, dim=1)
                z = torch.cat([zc, zs], dim=1)
                out = self.classifier(z)
                batch_y = batch_y.type(torch.LongTensor).to(device)

                loss = (self.loss(out, batch_y)
                        + config['training']['beta1'] * loss_con
                        + config['training']['beta2'] * loss_spe)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            #     all_icl += loss_icl.item()
            #     all_ccl += loss_ccl.item()
            #     all0 += all_loss.item()
            #     all1 += recon1.item()
            #     all2 += recon2.item()
            #     map1 += pre1.item()
            #     map2 += pre2.item()
            # output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
            #          "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> Loss_icl = {:.4e} ===> Los_ccl = {:.4e} ===> All loss = {:.4e}" \
            #     .format((epoch + 1), epochs, all1, all2, map1, map2, all_icl, all_ccl, all0)

            # if (epoch + 1) % config['print_num'] == 0:
            #     logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (epoch + 1) % config['print_num'] == 0:
                # save requires_grad state
                self.c_requires_grad_state = self.c.requires_grad
                # set requires_grad = False
                self.c.requires_grad_(False)

                with (torch.no_grad()):
                    for v in range(self.view_num):
                        self.autoencoders[v].eval(), self.f[v].eval()
                    self.fc.eval(), self.classifier.eval()

                    # Training data
                    # n_train = x_train[0].shape[0]
                    # z_s = []
                    # z_c = self.fc.encoder(self.c[index_train]).to(device)
                    #
                    # for v in range(self.view_num):
                    #     z_s_v = self.autoencoders[v].encoder(x_train[v])
                    #     z_s.append(z_s_v)

                    # z_s = torch.cat(z_s, dim=1)
                    # z_train = torch.cat([z_c, z_s], dim=1)

                    # Test data
                    # n_test = x_test[0].shape[0]
                    z_s = []
                    z_c = self.fc.encoder(self.c[index_test]).to(device)

                    for v in range(self.view_num):
                        z_s_v = self.autoencoders[v].encoder(x_test[v])
                        z_s.append(z_s_v)

                    z_s = torch.cat(z_s, dim=1)
                    z_test = torch.cat([z_c, z_s], dim=1)

                    # label_pre = classify.ave(z_train.cpu().numpy(), z_test.cpu().numpy(), y_train)
                    out = self.classifier(z_test)
                    label_pre = torch.max(out, dim=1)[1].cpu().numpy()

                    scores = accuracy_score(y_test, label_pre)

                    precision = precision_score(y_test, label_pre, average='macro', zero_division=1)
                    precision = np.round(precision, 2)

                    f_score = f1_score(y_test, label_pre, average='macro')
                    f_score = np.round(f_score, 2)

                    accumulated_metrics['acc'].append(scores)
                    accumulated_metrics['precision'].append(precision)
                    accumulated_metrics['f_measure'].append(f_score)
                    logger.info('\033[2;29m Accuracy on the test set is {:.4f}'.format(scores))
                    logger.info('\033[2;29m Precision on the test set is {:.4f}'.format(precision))
                    logger.info('\033[2;29m F_score on the test set is {:.4f}'.format(f_score))

                    for v in range(self.view_num):
                        self.autoencoders[v].train(), self.f[v].train()
                    self.fc.train(), self.classifier.train()
                    # restore requires_grad state
                    self.c.requires_grad_(self.c_requires_grad_state)

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], \
            accumulated_metrics['f_measure'][-1]

    def train_con_spe_accelerate(self, config, logger, accumulated_metrics, x, y, index_train, index_test, device,
                                 optimizer):
        x_train = []
        x_test = []

        for i in range(self.view_num):
            x_train_temp = torch.Tensor(x[i][index_train]).to(device)
            x_train.append(x_train_temp)
            x_test_temp = torch.Tensor(x[i][index_test]).to(device)
            x_test.append(x_test_temp)
        y_train = y[index_train]
        y_test = y[index_test]
        c_train = self.c[index_train]
        dataset = matDataset(x_train, y_train)
        train_loader = Data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
        epochs_total = config['training']['epoch']

        for epoch in range(epochs_total):
            # all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0
            for batch_idx, (batch_x, batch_y, idx) in enumerate(train_loader):
                # consistency
                for v in range(self.view_num):
                    batch_x[v] = batch_x[v].to(torch.float32).to(device)
                batch_c = c_train[idx].to(device)
                c_mu = batch_c.mean(dim=0)
                c_sigma = batch_c.std(dim=0)
                zc = self.fc.encoder(batch_c)
                zc_mu = zc.mean(dim=0)
                zc_sigma = zc.std(dim=0)
                I_zc_c = 0.5 * kl_div(c_mu, c_sigma, zc_mu, zc_sigma) + 0.5 * kl_div(zc_mu, zc_sigma, c_mu, c_sigma)
                # 将 batch_y 转换为 torch.float32 类型
                batch_y_float = batch_y.to(torch.float32).unsqueeze(dim=1).to(device)
                I_zc_y = mut_info_y(zc, batch_y.to(torch.int64).to(device), config['class_num'], device)
                loss_mse = 0
                for v in range(self.view_num):
                    c_v = self.f[v].encoder(batch_x[v])
                    loss_mse += F.mse_loss(c_v, batch_c)
                loss_con = (- cal_entropy(c_sigma.to('cpu')) + loss_mse - I_zc_y + config['training'][
                    'lambda1'] * I_zc_c)
                # loss_con = (- reyi_entropy(x, sigma) + loss_mse - I_zc_y + config['training']['lambda1'] * I_zc_c)

                # print(loss_con)

                #  specificity
                zs = []
                I_zs_y = 0
                I_zs_x = 0
                I_zs_zc = 0
                I_zsi_zj = 0
                for v in range(self.view_num):
                    zs_v = self.autoencoders[v].encoder(batch_x[v])
                    zs.append(zs_v)
                    I_zs_y += mut_info_y(zs_v, batch_y.to(torch.int64).to(device), config['class_num'], device)
                    zsv_mu = zs_v.mean()
                    zsv_sigma = zs_v.std()
                    I_zs_x += kl_norm(zsv_mu, zsv_sigma)
                    I_zs_zc += mut_info(zs_v, zc, device)

                for vi in range(self.view_num):
                    for vj in range(self.view_num):
                        if vi != vj:
                            I_zsi_zj += mut_info(zs[vi], zs[vj], device)

                loss_spe = - I_zs_y + config['training']['lambda2'] * I_zs_x + I_zs_zc + I_zsi_zj

                zs = torch.cat(zs, dim=1)
                z = torch.cat([zc, zs], dim=1)
                out = self.classifier(z)
                batch_y = batch_y.type(torch.LongTensor).to(device)

                loss = (self.loss(out, batch_y)
                        + config['training']['beta1'] * loss_con
                        + config['training']['beta2'] * loss_spe)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            #     all_icl += loss_icl.item()
            #     all_ccl += loss_ccl.item()
            #     all0 += all_loss.item()
            #     all1 += recon1.item()
            #     all2 += recon2.item()
            #     map1 += pre1.item()
            #     map2 += pre2.item()
            # output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
            #          "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> Loss_icl = {:.4e} ===> Los_ccl = {:.4e} ===> All loss = {:.4e}" \
            #     .format((epoch + 1), epochs, all1, all2, map1, map2, all_icl, all_ccl, all0)

            # if (epoch + 1) % config['print_num'] == 0:
            #     logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (epoch + 1) % config['print_num'] == 0:
                # save requires_grad state
                self.c_requires_grad_state = self.c.requires_grad
                # set requires_grad = False
                self.c.requires_grad_(False)

                with (torch.no_grad()):
                    for v in range(self.view_num):
                        self.autoencoders[v].eval(), self.f[v].eval()
                    self.fc.eval(), self.classifier.eval()

                    # Test data
                    z_s = []
                    z_c = self.fc.encoder(self.c[index_test]).to(device)

                    for v in range(self.view_num):
                        z_s_v = self.autoencoders[v].encoder(x_test[v])
                        z_s.append(z_s_v)

                    z_s = torch.cat(z_s, dim=1)
                    z_test = torch.cat([z_c, z_s], dim=1)

                    # label_pre = classify.ave(z_train.cpu().numpy(), z_test.cpu().numpy(), y_train)
                    out = self.classifier(z_test)
                    label_pre = torch.max(out, dim=1)[1].cpu().numpy()

                    scores = accuracy_score(y_test, label_pre)

                    precision = precision_score(y_test, label_pre, average='macro', zero_division=1)
                    precision = np.round(precision, 2)

                    f_score = f1_score(y_test, label_pre, average='macro')
                    f_score = np.round(f_score, 2)

                    accumulated_metrics['acc'].append(scores)
                    accumulated_metrics['precision'].append(precision)
                    accumulated_metrics['f_measure'].append(f_score)
                    logger.info('\033[2;29m Accuracy on the test set is {:.4f}'.format(scores))
                    logger.info('\033[2;29m Precision on the test set is {:.4f}'.format(precision))
                    logger.info('\033[2;29m F_score on the test set is {:.4f}'.format(f_score))

                    for v in range(self.view_num):
                        self.autoencoders[v].train(), self.f[v].train()
                    self.fc.train(), self.classifier.train()
                    # restore requires_grad state
                    self.c.requires_grad_(self.c_requires_grad_state)

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], \
            accumulated_metrics['f_measure'][-1]
