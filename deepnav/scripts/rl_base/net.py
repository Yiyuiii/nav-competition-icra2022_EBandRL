import numpy as np
import torch
import torch.nn as nn

class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


def get_net_cnn(out_dim):
    return nn.Sequential(  # size==(batch_size, 1, 200, 200)
        nn.Conv2d(1, 8, (5, 5), stride=(2, 2), bias=False), nn.ReLU(inplace=True),  # size=100
        nn.Conv2d(8, 16, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=50
        nn.Conv2d(16, 32, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=25
        nn.Conv2d(32, 64, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=12
        nn.Conv2d(64, 128, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=6
        nn.Conv2d(128, 256, (5, 5), stride=(1, 1)), nn.ReLU(inplace=True),  # size=1
        NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
        nn.Linear(256, out_dim),  # size==(batch_size, out_dim)
    )

def get_net_mlp(input_dim, mid_dim, activation):
    return nn.Linear(input_dim, mid_dim), activation(), nn.Linear(mid_dim, mid_dim), nn.Hardswish()


class AAT(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.a_std_log_shift = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)

    def forward(self):
        return self.a_std_log_shift


class ActorCNN(nn.Module):
    """
    input: [state, state1d]
    """

    def __init__(self,
                 mid_dim,
                 state_dim,
                 action_dim,
                 activation,
                 cnn,
                 cnn_out_dim,
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.cnn = cnn
        assert isinstance(state_dim, int)

        self.net = nn.Sequential(*get_net_mlp(state_dim + cnn_out_dim, mid_dim, activation),
                        nn.Linear(mid_dim, action_dim * 2), )
        self.weights_init()

        #  Trainable std, same to Automatically Adjusted Temperature in SAC
        self.a_std_log_shift = AAT(action_dim)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def weights_init(self):
        for m in self.net.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                layer_norm(m, std=1.41)
        layer_norm(self.net[-1], std=0.01)  # output layer for action

    def forward(self, state, state1d):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        a_avg, a_std, a_noise, logprob = self.get_action(state, state1d)
        return a_avg  # .tanh()

    def get_action(self, state, state1d):
        """
        The forward function with Gaussian noise.

        :param state[np.array]: the input state.
        :return: the action and added noise.
        """
        state = torch.unsqueeze(state, dim=1)
        emb = self.cnn(state)
        emb = torch.cat((emb, state1d), dim=1)

        a = self.net(emb.detach())
        a_avg, a_std_log = torch.split(a, self.action_dim, dim=1)
        a_std_log = (a_std_log + self.a_std_log_shift()).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std

        logprob = -(a_std_log + self.sqrt_2pi_log + noise.pow(2) * 0.5).sum(1)
        return a_avg, a_std, a_noise, logprob

    def get_logprob_entropy(self, state, action, state1d):
        """
        Compute the log of probability with current network.

        :param state[np.array]: the input state.
        :param action[float]: the action.
        :return: the log of probability and entropy.
        """
        state = torch.unsqueeze(state, dim=1)
        emb = self.cnn(state)
        emb = torch.cat((emb, state1d), dim=1)

        a = self.net(emb.detach())
        a_avg, a_std_log = torch.split(a, self.action_dim, dim=1)
        a_std_log = (a_std_log + self.a_std_log_shift()).clamp(-20, 2)

        # action = action.atanh()
        delta = ((a_avg - action) / a_std_log.exp()).pow(2) * 0.5
        logprob = -(a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_modules(self):
        return self.net, self.a_std_log_shift


class CriticCNN(nn.Module):
    """
    The Critic class for **PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, _action_dim, activation, cnn, cnn_out_dim):
        super().__init__()
        self.cnn = cnn
        self.net = nn.Sequential(*get_net_mlp(state_dim + cnn_out_dim, mid_dim, activation),
                                 nn.Linear(mid_dim, 1), )
        self.weights_init()

    def weights_init(self):
        for m in self.net.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                layer_norm(m, std=1.41)
        layer_norm(self.net[-1], std=1.0)  # output layer for adv

    def forward(self, state, state1d):
        """
        The forward function to ouput the value of the state.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        state = torch.unsqueeze(state, dim=1)
        emb = self.cnn(state)
        emb = torch.cat((emb, state1d), dim=1)
        return self.net(emb)   # advantage value

    def get_modules(self):
        return self.net, self.cnn


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
