import torch
import math
import time
from utils.model_MI import Encoder, MIEstimator, MIEstimator_y


def cal_entropy(sigma_c):
    epsilon = 1e-6
    H_C = torch.log(sigma_c + epsilon) + 1/2*torch.log(torch.Tensor([2*math.pi*math.e]))
    return H_C.mean()


def pairwise_distances(x):
    bn = x.shape[0]
    x = x.view(bn, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def kl_div(mu_q, std_q, mu_p, std_p):
    """Computes the KL divergence between the two given variational distribution.\
       This computes KL(gllp), which is not symmetric. It quantifies how far is\
       The estimated distribution q from the true distribution of p.
       计算变分后验和先验之间的 KL 散度，这是一种衡量一个概率分布与第二个期望概率分布偏离程度的方法。"""
    k = len(mu_q) # ！这里有可能会报错
    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    dims = std_p.ndim-1
    logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=dims)
    logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=dims)
    fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=dims) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=dims)
    kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
    return kl_divergence.mean()


def kl_norm(mu, std):
    # natural_log_of_2 = torch.log(torch.tensor(2.0))
    natural_log_of_2 = 0.69314718
    info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(std.ndim-1).mean().div(natural_log_of_2)
    return info_loss


def mut_info(v1, v2, device):
    encoder_v1 = Encoder(v1.shape[1]).to(device)
    encoder_v2 = Encoder(v2.shape[1]).to(device)
    p_z1_given_v1 = encoder_v1(v1)
    p_z2_given_v2 = encoder_v2(v2)

    # Sample from the posteriors with reparametrization
    z1 = p_z1_given_v1.rsample()
    z2 = p_z2_given_v2.rsample()

    mi_estimator = MIEstimator(v1.shape[1], v2.shape[1]).to(device)
    mi_gradient, mi_estimation = mi_estimator(z1, z2)
    mi_gradient = mi_gradient.mean()
    return mi_gradient


def mut_info_y(z, y, y_num, device):
    input_dim = z.shape[1]  # dimension  of z
    output_dim = y_num  # class number
    model = MIEstimator_y(input_dim, output_dim).to(device)

    # 计算 I(Z_c; Y)
    mi = model(z, y)
    return mi
