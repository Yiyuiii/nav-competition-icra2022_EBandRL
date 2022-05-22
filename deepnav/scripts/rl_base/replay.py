import os
import torch
import numpy as np
import numpy.random as rd
from einops import rearrange


class ReplayBuffer:
    def __init__(self,
                 max_len,
                 state_dim,
                 other_dict=None,
                 if_use_per=False,
                 if_use_gae=False,
                 if_use_hidden=False,
                 lambda_gae_adv=0.98,
                 lambda_a_value=1.00,
                 gpu_id=0):
        """Experience Replay Buffer

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        `int max_len` the maximum capacity of ReplayBuffer. First In First Out
        `int state_dim` the dimension of state
        `int action_dim` the dimension of action (action_dim==1 for discrete action)
        `bool if_on_policy` on-policy or off-policy
        `bool if_per` Prioritized Experience Replay for sparse reward

        Input traj is a dict whose key fits other_dict
        """
        if other_dict is None:
            other_dict = {'reward': 1, 'done': 1, 'action': 1}
        self.other_dict = other_dict
        self.other_dict_name = other_dict.keys()
        self.other_dim_list = list(other_dict.values())

        self.if_use_per = if_use_per
        self.if_use_gae = if_use_gae
        self.if_use_hidden = if_use_hidden
        self.lambda_gae_adv = lambda_gae_adv
        self.lambda_a_value = lambda_a_value
        if if_use_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.max_len = max_len
        self.state_dim = state_dim
        self.data_type = torch.float32
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.init_buffer()
        self.reset()

    def init_buffer(self):
        self.per_tree = False
        other_dim = np.sum(self.other_dim_list)
        self.buf_other = torch.empty((self.max_len, other_dim), dtype=torch.float32, device=self.device)

        if isinstance(self.state_dim, int):  # state is pixel
            self.buf_state = torch.empty((self.max_len, self.state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(self.state_dim, tuple):
            self.buf_state = torch.empty((self.max_len, *self.state_dim), dtype=torch.float32, device=self.device)
        else:
            raise ValueError('state_dim')

        self.buf_r_sum = torch.empty(self.max_len, dtype=torch.float32, device=self.device)  # old policy value
        self.buf_adv_v = torch.empty(self.max_len, dtype=torch.float32, device=self.device)  # advantage value

    def reset(self):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False

    def other_dict2tensor(self, other):
        buf_other = []
        for name in self.other_dict_name:
            buf_other.append(other[name])
        return torch.cat(buf_other, dim=1)

    def other_tensor2dict(self, other):
        other_dict = {}
        p = 0
        for name in self.other_dict_name:
            p_ = p + self.other_dict[name]
            other_dict[name] = other[:, p:p_]
            p = p_
        return other_dict

    def append_sample(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = self.other_dict2tensor(other)

        if self.per_tree:
            self.per_tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def append_traj(self, state, _other):  # CPU array to CPU array
        other = self.other_dict2tensor(_other)
        size = len(other)
        next_idx = self.next_idx + size

        if self.per_tree:
            self.per_tree.update_ids(data_ids=np.arange(self.next_idx, next_idx) % self.max_len)

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        if self.per_tree:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)
            other = self.buf_other[indices]
            other_dict = self.other_tensor2dict(other)
            return (self.buf_state[indices].type(torch.float32),  # state
                    self.buf_state[indices + 1].type(torch.float32),  # next state
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device),  # important sampling weights
                    *list(other_dict.values()),
                    )
        else:
            indices = rd.randint(self.now_len - 1, size=batch_size)
            other = self.buf_other[indices]
            other_dict = self.other_tensor2dict(other)
            return (self.buf_state[indices],
                    self.buf_state[indices + 1],
                    *list(other_dict.values()),
                    self.buf_r_sum[indices],
                    self.buf_adv_v[indices],
                    )

    def update_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def cal_adv(self, cri, cri_device):
        with torch.no_grad():
            other_dict = self.other_tensor2dict(self.buf_other[:self.now_len])
            '''get buf_r_sum, buf_logprob'''
            bs = 1024
            if self.if_use_hidden:
                buf_value = list()
                for i in range(0, self.now_len, bs):
                    i_ = min(i + bs, self.now_len)
                    buf_value.append(cri(self.buf_state[i:i_].to(cri_device), other_dict['hidden'][i:i_].to(cri_device)))
            else:
                buf_value = [cri(self.buf_state[i:i + bs].to(cri_device)) for i in range(0, self.now_len, bs)]
            buf_value = torch.cat(buf_value, dim=0).squeeze().to(self.device)


            self.get_reward_sum(other_dict['reward'].squeeze(), other_dict['done'].squeeze(), buf_value[:self.now_len])  # detach()
            '''adv normalization'''
            adv_v = self.buf_adv_v[:self.now_len]
            self.buf_adv_v[:self.now_len] = (adv_v - adv_v.mean()) * (self.lambda_a_value / (adv_v.std() + 1e-5))

    def get_reward_sum_raw(self, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        pre_r_sum = 0
        buf_reward_list = torch.unbind(buf_reward)
        buf_mask_list = torch.unbind(buf_mask)
        for i in range(self.now_len - 1, -1, -1):
            pre_r_sum = buf_reward_list[i] + buf_mask_list[i] * pre_r_sum
            self.buf_r_sum[i] = pre_r_sum
        self.buf_adv_v[:self.now_len] = self.buf_r_sum[:self.now_len] - (buf_mask * buf_value)  # buf_advantage_value

    def get_reward_sum_gae(self, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        buf_reward_list = torch.unbind(buf_reward)
        buf_mask_list = torch.unbind(buf_mask)
        buf_value_list = torch.unbind(buf_value)
        for i in range(self.now_len - 1, -1, -1):
            pre_r_sum = buf_reward_list[i] + buf_mask_list[i] * pre_r_sum
            self.buf_r_sum[i] = pre_r_sum
            buf_adv_v = buf_reward_list[i] - buf_value_list[i] + buf_mask_list[i] * pre_adv_v  # fix a bug here
            self.buf_adv_v[i] = buf_adv_v
            pre_adv_v = buf_value_list[i] + buf_adv_v * self.lambda_gae_adv

    def td_error_update(self, td_error):
        self.per_tree.td_error_update(td_error)

    # TODO
    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        save_path = f"{cwd}/replay_{buffer_id}.npz"
        if_load = None

        if if_save:
            self.update_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]

            buf_state_data_type = np.float16 \
                if self.buf_state.dtype in {np.float, np.float64, np.float32} \
                else np.uint8

            buf_state = np.empty((self.max_len, state_dim), dtype=buf_state_data_type)
            buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

            temp_len = self.max_len - self.now_len
            buf_state[0:temp_len] = self.buf_state[self.now_len:self.max_len].detach().cpu().numpy()
            buf_other[0:temp_len] = self.buf_other[self.now_len:self.max_len].detach().cpu().numpy()

            buf_state[temp_len:] = self.buf_state[:self.now_len].detach().cpu().numpy()
            buf_other[temp_len:] = self.buf_other[:self.now_len].detach().cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict['buf_state']
            buf_other = buf_dict['buf_other']

            buf_state = torch.as_tensor(buf_state, dtype=torch.float32, device=self.device)
            buf_other = torch.as_tensor(buf_other, dtype=torch.float32, device=self.device)
            self.extend_buffer(buf_state, buf_other)
            self.update_len()
            print(f"| ReplayBuffer load: {save_path}")
            if_load = True
        else:
            # print(f"| ReplayBuffer FileNotFound: {save_path}")
            if_load = False
        return if_load


class ReplayBufferMP(ReplayBuffer):
    def __init__(self,
                 max_len,
                 env_num,
                 state_dim,
                 other_dict=None,
                 if_use_per=False,
                 if_use_gae=False,
                 if_use_hidden=False,
                 lambda_gae_adv=0.98,
                 lambda_a_value=1.00,
                 gpu_id=0):
        """Experience Replay Buffer

        In order to maintain env track integrity under async sampling.

        `int max_len` the maximum capacity of ReplayBuffer. First In First Out
        `int state_dim` the dimension of state
        `int action_dim` the dimension of action (action_dim==1 for discrete action)
        `bool if_on_policy` on-policy or off-policy
        `bool if_per` Prioritized Experience Replay for sparse reward

        Input traj is a dict whose key fits other_dict
        """
        self.env_num = env_num
        ReplayBuffer.__init__(self,
                              max_len=max_len,
                              state_dim=state_dim,
                              other_dict=other_dict,
                              if_use_per=if_use_per,
                              if_use_gae=if_use_gae,
                              if_use_hidden=if_use_hidden,
                              lambda_gae_adv=lambda_gae_adv,
                              lambda_a_value=lambda_a_value,
                              gpu_id=gpu_id)

    def init_buffer(self):
        other_dim = np.sum(self.other_dim_list)
        self.buf_other = torch.empty((self.max_len, self.env_num, other_dim), dtype=torch.float32, device=self.device)

        if isinstance(self.state_dim, int):  # state is pixel
            self.buf_state = torch.empty((self.max_len, self.env_num, self.state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(self.state_dim, tuple):
            self.buf_state = torch.empty((self.max_len, self.env_num, *self.state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

        self.buf_r_sum = torch.empty(self.max_len, self.env_num, dtype=torch.float32, device=self.device)  # old policy value
        self.buf_adv_v = torch.empty(self.max_len, self.env_num, dtype=torch.float32, device=self.device)  # advantage value

    def reset(self):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        if self.if_use_per:
            self.per_tree = [BinarySearchTree(self.max_len) for _ in range(self.env_num)]

    def other_dict2tensor(self, other):
        """(s, e, d)"""
        buf_other = []
        for name in self.other_dict_name:
            buf_other.append(other[name])
        return torch.cat(buf_other, dim=2)

    def other_tensor2dict(self, other):
        other_dict = {}
        p = 0
        for name in self.other_dict_name:
            p_ = p + self.other_dict[name]
            other_dict[name] = other[:, :, p:p_]
            p = p_
        return other_dict

    def append_sample(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = self.other_dict2tensor(other)

        if self.if_use_per:
            for per_tree in self.per_tree:
                per_tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def append_traj(self, state, _other):  # CPU array to CPU array
        """(s, e, d)"""
        other = self.other_dict2tensor(_other)
        size = len(other)
        next_idx = self.next_idx + size

        if self.if_use_per:
            for per_tree in self.per_tree:
                per_tree.update_ids(data_ids=np.arange(self.next_idx, next_idx) % self.max_len)

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[:next_idx] = state[-next_idx:]
            self.buf_other[:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        if self.if_use_per:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices = list()
            is_weights = list()
            bs = np.ceil(batch_size / self.env_num)
            for i_env, per_tree in enumerate(self.per_tree):
                indice, is_weight = per_tree.get_indices_is_weights(bs, beg, end)
                indices.append(np.stack([indice, np.ones_like(indice) * i_env], axis=1))
                is_weights.append(is_weight)
            indices = np.concatenate(indices, axis=0)
            is_weights = np.concatenate(is_weights, axis=0)
            i_i = rd.randint(len(is_weights)-1, size=batch_size)
            indices = indices[i_i]
            is_weights = is_weights[i_i]

            other = self.buf_other[indices]
            other_dict = self.other_tensor2dict(other)
            return (self.buf_state[indices].type(torch.float32),  # state
                    self.buf_state[indices + (1, 0)].type(torch.float32),  # next state
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device),  # important sampling weights
                    *list(other_dict.values()),
                    )
        else:
            indices = rd.randint(self.now_len * self.env_num - 1, size=batch_size)
            i_pos = indices % self.now_len
            i_env = indices // self.now_len

            other = self.buf_other[i_pos, i_env]
            other_dict = ReplayBuffer.other_tensor2dict(self, other)
            return (self.buf_state[i_pos, i_env],
                    self.buf_state[i_pos+1, i_env],
                    *list(other_dict.values()),
                    self.buf_r_sum[i_pos, i_env],
                    self.buf_adv_v[i_pos, i_env],
                    )

    def cal_adv(self, cri, cri_device):
        with torch.no_grad():
            other_dict = self.other_tensor2dict(self.buf_other[:self.now_len])
            '''get buf_r_sum, buf_logprob'''
            bs = 1024
            buf_state = rearrange(self.buf_state[:self.now_len], 's e d -> (s e) d')
            buf_value = [cri(buf_state[i:i + bs].to(cri_device)) for i in range(0, buf_state.shape[0], bs)]
            buf_value = torch.cat(buf_value, dim=0).squeeze().to(self.device)
            buf_value = rearrange(buf_value, '(s e) -> s e', e=self.env_num)

            self.get_reward_sum(other_dict['reward'].squeeze(), other_dict['done'].squeeze(), buf_value)  # detach()
            '''adv normalization'''
            adv_v = self.buf_adv_v[:self.now_len]
            self.buf_adv_v[:self.now_len] = (adv_v - adv_v.mean()) * (self.lambda_a_value / (adv_v.std() + 1e-5))

    def get_reward_sum_raw(self, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        pre_r_sum = 0
        buf_reward_list = torch.unbind(buf_reward)
        buf_mask_list = torch.unbind(buf_mask)
        for i in range(self.now_len - 1, -1, -1):
            pre_r_sum = buf_reward_list[i] + buf_mask_list[i] * pre_r_sum
            self.buf_r_sum[i] = pre_r_sum
        self.buf_adv_v[:self.now_len] = self.buf_r_sum[:self.now_len] - (buf_mask * buf_value)  # buf_advantage_value

    def get_reward_sum_gae(self, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        buf_reward_list = torch.unbind(buf_reward)
        buf_mask_list = torch.unbind(buf_mask)
        buf_value_list = torch.unbind(buf_value)

        for i in range(self.now_len - 1, -1, -1):
            pre_r_sum = buf_reward_list[i] + buf_mask_list[i] * pre_r_sum
            self.buf_r_sum[i] = pre_r_sum
            buf_adv_v = buf_reward_list[i] - buf_value_list[i] + buf_mask_list[i] * pre_adv_v  # fix a bug here
            self.buf_adv_v[i] = buf_adv_v
            pre_adv_v = buf_value_list[i] + buf_adv_v * self.lambda_gae_adv

    def td_error_update(self, td_error_list):
        for i, per_tree in enumerate(self.per_tree):
            per_tree.td_error_update(td_error_list[i])

