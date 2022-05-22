import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
import os
from rl_base.net import ActorCNN, CriticCNN, get_net_cnn

class AgentBase:
    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        """initialize
        replace by different DRL algorithms
        explict call self.init() for multiprocessing.
        `net_dim` the dimension of networks (the width of neural networks)
        `state_dim` the dimension of state (the number of state vector)
        `action_dim` the dimension of action (the number of discrete action)
        `learning_rate` learning rate of optimizer
        `if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `gpu_id` the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.states = None
        self.device = None
        self.traj_list = None
        self.action_dim = None
        self.if_off_policy = True

        self.env_num = 1
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0
        # self.amp_scale = None  # automatic mixed precision

        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """initialize the self.object in `__init__()`
        replace by different DRL algorithms
        explict call self.init() for multiprocessing.
        `net_dim` the dimension of networks (the width of neural networks)
        `state_dim` the dimension of state (the number of state vector)
        `action_dim` the dimension of action (the number of discrete action)
        `learning_rate` learning rate of optimizer
        `if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `gpu_id` the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.action_dim = action_dim
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.traj_list = [list() for _ in range(env_num)]
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer
        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning
        `buffer` Experience replay buffer.
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        `float repeat_times` the times of sample batch = int(target_step * repeat_times) in off-policy
        `float soft_update_tau` target_net = target_net * (1-tau) + current_net * tau
        `return tuple` training logging. tuple = (float, float, ...)
        """

    def optim_update(self, optimizer, objective, params):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()

    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
    #     # self.amp_scale = torch.cuda.amp.GradScaler()
    #
    #     optimizer.zero_grad()
    #     self.amp_scale.scale(objective).backward()  # loss.backward()
    #     self.amp_scale.unscale_(optimizer)  # amp
    #
    #     # from torch.nn.utils import clip_grad_norm_
    #     # clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
    #     self.amp_scale.step(optimizer)  # optimizer.step()
    #     self.amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network
        `nn.Module target_net` target network update via a current network, it is more stable
        `nn.Module current_net` current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load the training files for agent from disk.
        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                if os.path.isfile(save_path):
                    load_torch_file(obj, save_path)
                else:
                    print(f'| Not found {save_path}.')

    @staticmethod
    def convert_trajectory(traj_list, reward_scale, gamma):  # off-policy
        for ten_state, ten_other in traj_list:
            ten_other[:, 0] = ten_other[:, 0] * reward_scale  # ten_reward
            ten_other[:, 1] = (1.0 - ten_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma
        return traj_list

class CnnPPO(AgentBase):
    """
    Bases: ``elegantrl.agent.AgentBase``

    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """
    def __init__(self, if_on_policy=True):
        AgentBase.__init__(self)
        self.if_on_policy = if_on_policy
        self.ClassAct = ActorCNN
        self.ClassCri = CriticCNN

    def init(self,
             state_dim=8,
             action_dim=2,
             net_dim=512,
             activation=torch.nn.ReLU,
             optimizer=torch.optim.Adam,
             lr_scheduler=torch.optim.lr_scheduler.CyclicLR,
             lr_scheduler_kwargs=dict(),
             lr_cri=3e-4,
             lr_act=3e-4,
             device="cpu",
             ):
        """initialize t
        he self.object in `__init__()`
        replace by different DRL algorithms
        explict call self.init() for multiprocessing.
        `net_dim` the dimension of networks (the width of neural networks)
        `state_dim` the dimension of state (the number of state vector)
        `action_dim` the dimension of action (the number of discrete action)
        `learning_rate` learning rate of optimizer
        `if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `gpu_id` the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.action_dim = action_dim
        self.net_dim = net_dim
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.device = device

        cnn_out_dim = net_dim
        self.cnn = get_net_cnn(out_dim=cnn_out_dim)
        self.cri = self.ClassCri(net_dim, state_dim, action_dim, activation, self.cnn, cnn_out_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim, activation, self.cnn, cnn_out_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act
        self.cri_optim = optimizer([{"params":m.parameters()} for m in self.cri.get_modules()], lr=lr_cri)
        self.act_optim = optimizer([{"params":m.parameters()} for m in self.act.get_modules()], lr=lr_act) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

        # self.cri_lr_scheduler = lr_scheduler(self.cri_optim, **lr_scheduler_kwargs)
        # self.act_lr_scheduler = lr_scheduler(self.act_optim, **lr_scheduler_kwargs)

    def init_static_param(self,
                          if_per_or_gae=False,
                          ratio_clip=0.2,
                          lambda_entropy=0.02,
                          lambda_a_value=1.00,
                          lambda_gae_adv=0.98,
                          ):
        self.ratio_clip = ratio_clip  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = lambda_entropy  # could be 0.00~0.10
        self.lambda_a_value = lambda_a_value  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = lambda_gae_adv  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)

        if if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def select_action(self, state, state1d) -> tuple:
        s_tensor = torch.as_tensor(state[np.newaxis], dtype=torch.float32, device=self.device)
        h_tensor = torch.as_tensor(state1d[np.newaxis], dtype=torch.float32, device=self.device)
        a_avg, a_std, a_noise, logprob = self.act.get_action(s_tensor, h_tensor)
        a_avg = a_avg[0]
        a_std = a_std[0]
        a_noise = a_noise[0]
        logprob = logprob[0]
        return a_avg.detach().cpu(), a_std.detach().cpu(), a_noise.detach().cpu(), logprob.detach().cpu()

    def select_actions(self, state, state1d) -> tuple:
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        h_tensor = torch.as_tensor(state1d, dtype=torch.float32, device=self.device)
        a_avg, a_std, a_noise, logprob = self.act.get_action(s_tensor, h_tensor)
        return a_avg.detach().cpu(), a_std.detach().cpu(), a_noise.detach().cpu(), logprob.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """
        for repeat_i in range(0, repeat_times):
            buffer.cal_adv(self.cri_target, self.device)
            state, state_, action, reward, done, logprob, hidden, r_sum, adv_v = buffer.sample_batch(batch_size)
            state = state.to(self.device)
            action = action.to(self.device)
            logprob = logprob.to(self.device).squeeze()
            r_sum = r_sum.to(self.device)
            adv_v = adv_v.to(self.device)
            hidden = hidden.to(self.device)

            '''PPO: Surrogate objective of Trust Region'''
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action, hidden)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state, hidden).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            self.act_optim.zero_grad()
            self.cri_optim.zero_grad()
            obj_actor.backward()
            obj_critic.backward()
            clip_grad_norm_(self.act.parameters(), max_norm=self.clip_grad_norm)
            clip_grad_norm_(self.cri.parameters(), max_norm=self.clip_grad_norm)
            self.act_optim.step()
            self.cri_optim.step()
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None
            # self.cri_lr_scheduler.step()
            # self.act_lr_scheduler.step()

        return obj_critic.item(), obj_actor.item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - (buf_mask * buf_value[:, 0])  # buf_advantage_value
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * (pre_adv_v - ten_value[i])  # fix a bug here
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
        return buf_r_sum, buf_adv_v

    def save_or_load_agent(self, cwd, if_save):
        """save or load the training files for agent from disk.
        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim),]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                if os.path.isfile(save_path):
                    load_torch_file(obj, save_path)
                else:
                    print(f'| Not found {save_path}.')
            self.act.cnn = self.cri.cnn
            print(f'Agent loaded.')
