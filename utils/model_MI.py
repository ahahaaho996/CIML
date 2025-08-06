import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import Normal, Independent
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class Decoder(nn.Module):
    def __init__(self, z_dim, scale=0.39894):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.scale = scale

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28)
        )

    def forward(self, z):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)


class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


class MIEstimator_y(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MIEstimator_y, self).__init__()
        # define a linear layer, to generate q(y|z_c)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, z_c, y_true):
        """
        Calculate the variational lower bound of I (Z_c; Y)
        :param z_c: (batch_size, input_dim) consistency representation
        :param y_true: (batch_size,) true label
        :return: estimation of I(Z_c; Y)
        """
        # The unnormalized probability is obtained by passing the linear layer
        logits = self.linear(z_c)
        # Use softmax to transform to normalized probability distribution q(y|z_c)
        q_y_given_zc = F.softmax(logits, dim=1)

        # Compute the log-likelihood under q(y|z_c)
        log_q_y_given_zc = F.log_softmax(logits, dim=1)

        # Get the log-likelihood corresponding to y_true
        log_q_y_given_zc_for_y_true = log_q_y_given_zc.gather(1, y_true.unsqueeze(1)).squeeze()

        # Compute the log-likelihood of the empirical data distribution p(y)
        log_p_y = -torch.log(torch.tensor(q_y_given_zc.size(1), dtype=torch.float32))

        # Calculate the variational lower bound of I (Z_c; Y)
        mi_estimate = log_q_y_given_zc_for_y_true.mean() - log_p_y

        return mi_estimate


