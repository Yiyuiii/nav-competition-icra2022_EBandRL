import copy
import os
import time
import torch
import numpy as np

from .util import time_text, do_state_norm, set_seed
from gym import logger as gymlogger
gymlogger.set_level(40)


def log_hparam(args):
    import json

    def get_hparam_dict(hparam_dict, item, name):
        if isinstance(item, (bool, str, float, int)):
            hparam_dict[name] = item
        elif isinstance(item, (dict,)):
            for key in item:
                get_hparam_dict(hparam_dict, item[key], f'{name}.{key}')
        elif hasattr(item, '__name__'):
            hparam_dict[name] = item.__name__
        elif item is None:
            hparam_dict[name] = 'None'
        else:
            hparam_dict[name] = item.__class__.__name__

    hparam_dict = {}
    for name in args.__dict__:
        get_hparam_dict(hparam_dict, getattr(args, name), name)

    args.logger.add_hparams(hparam_dict=hparam_dict,
                            metric_dict={'hparam/sign': 1})
    hparam_json = json.dumps(hparam_dict, sort_keys=True, indent=4, separators=(',', ': '))
    with open(os.path.join(args.work_dir, "hparam.json"), "w") as f:
        f.write(hparam_json)
    del hparam_dict, hparam_json


def init_before_training(args, first_run=True):
    """some init"""
    import shutil
    from tensorboardX import SummaryWriter

    '''seeds'''
    args.random_seed = set_seed(seed=args.random_seed)
    torch.set_num_threads(args.thread_num)
    torch.set_default_dtype(torch.float32)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu

    '''env'''
    if args.env_class is None:
        raise RuntimeError('\n| Why env_class=None?')
    if args.env_is_vec:
        from envs.wrapper import get_env
        envs = get_env(args)
        envs.reset()
    else:
        '''parallel vec env'''
        from envs.wrapper import Env_process_wrapper
        envs = [Env_process_wrapper(args) for _ in range(args.worker_num)]

    '''agent'''
    if args.agent is None:
        raise RuntimeError(f'\n| Why agent=None? Assignment `args.agent = AgentXXX` please.')
    if not hasattr(args.agent, 'init'):
        raise RuntimeError(f"\n| why hasattr(self.agent, 'init') == False"
                           f'\n| Should be `agent=AgentXXX()` instead of `agent=AgentXXX`.')
    if args.agent.if_on_policy != args.if_on_policy:
        raise RuntimeError(f'\n| Why bool `if_on_policy` is not consistent?'
                           f'\n| self.if_on_policy: {args.if_on_policy}'
                           f'\n| self.agent.if_on_policy: {args.agent.if_on_policy}')
    args.agent.init(state_dim=args.state_dim,
                    action_dim=args.action_dim,
                    net_dim_act=args.net_dim_policy,
                    net_dim_cri=args.net_dim_critic,
                    activation=args.activation,
                    optimizer=args.optimizer,
                    lr_scheduler=args.lr_scheduler,
                    lr_scheduler_kwargs=args.lr_scheduler_kwargs,
                    lr_cri=args.learning_rate_critic,
                    lr_act=args.learning_rate_policy,
                    device=args.device,
                    )

    args.agent.init_static_param(**args.agent_kwargs)

    if args.agent_load_dir is not None:
        print('| Load existing agent.')
        args.agent.save_or_load_agent(cwd=args.agent_load_dir, if_save=False)

    '''replay'''
    args.buffer_key = {'a': 'action', 'r': 'reward', 'd': 'done', 'prob': 'logprob'}
    replay_dict = {'action': args.action_dim, 'reward': 1, 'done': 1, 'logprob': 1}
    if args.env_is_vec:
        from rl_base.replay import ReplayBufferMP
        args.buffer = ReplayBufferMP(max_len=args.buffer_maxlen,
                                     state_dim=args.state_dim,
                                     env_num=args.env_num,
                                     other_dict=replay_dict,
                                     if_use_per=False,
                                     if_use_gae=args.if_gae,
                                     lambda_gae_adv=args.lambda_gae_adv,
                                     lambda_a_value=args.lambda_a_value,
                                     gpu_id=args.buffer_gpu)
    else:
        from rl_base.replay import ReplayBuffer
        args.buffer = ReplayBuffer(max_len=args.buffer_maxlen,
                                   state_dim=args.state_dim,
                                   other_dict=replay_dict,
                                   if_use_per=False,
                                   if_use_gae=args.if_gae,
                                   lambda_gae_adv=args.lambda_gae_adv,
                                   lambda_a_value=args.lambda_a_value,
                                   gpu_id=args.buffer_gpu)

    '''work_dir check'''
    if args.work_dir is None:
        agent_name = args.agent.__class__.__name__
        # env_name = getattr(args.env, 'env_name', args.env)
        iter_num = 0
        args.work_dir = f'./log/{agent_name}_{args.env_name}_{args.random_seed}_{iter_num}'
        while os.path.exists(args.work_dir):
            iter_num += 1
            args.work_dir = f'./log/{agent_name}_{args.env_name}_{args.random_seed}_{iter_num}'
    if first_run:
        # remove history according to bool(if_remove)
        if args.if_clear_work_dir is None:
            args.if_clear_work_dir = bool(input(f"| PRESS 'y' to REMOVE: {args.work_dir}? ") == 'y')
        elif args.if_clear_work_dir:
            shutil.rmtree(args.work_dir, ignore_errors=True)
            print(f"| Remove dir: {args.work_dir}")
        os.makedirs(args.work_dir, exist_ok=True)
    args.logger = SummaryWriter(log_dir=args.work_dir)

    '''save hparam'''
    log_hparam(args)

    return envs


def update_buffer(args, buffer, traj_dict, key_dict, state_norm=None):
    _steps = traj_dict['r'].shape[0]
    state = traj_dict['s']
    norm_dims = tuple(range(len(state.shape)-1))
    if isinstance(state, torch.Tensor):
        state_mean = state.mean(dim=norm_dims).to(args.device)
        state_var = state.var(dim=norm_dims).to(args.device)
    else:
        state_mean = np.mean(state, axis=norm_dims)
        state_var = np.var(state, axis=norm_dims)

    if state_norm is not None:
        state_mean, state_var = state_norm.update(_steps, state_mean, state_var)
        if isinstance(state, torch.Tensor):
            state_mean = state_mean.cpu()
            state_var = state_var.cpu()
        state = do_state_norm(state, state_mean, state_var)
        traj_dict['s'] = state

    for k in traj_dict.keys():
        if not isinstance(traj_dict[k], torch.Tensor):
            traj_dict[k] = torch.as_tensor(traj_dict[k], dtype=torch.float32, device=buffer.device)

    traj_dict['r'] *= args.reward_scale
    traj_dict['d'] = (1.0 - traj_dict['d']) * args.gamma  # ten_mask = (1.0 - ary_done) * gamma
    others = dict()
    for k, v in key_dict.items():
        if k in ('r', 'd', 'prob'):
            traj_dict[k] = traj_dict[k].unsqueeze(dim=-1)
        others[v] = traj_dict[k]

    buffer.append_traj(traj_dict['s'], others)

    return _steps, state_mean, state_var


def explore_issacgym(args, envs, agent=None, target_step=None, target_turn=None, state_norm=None, rand_act=True):
    """
    Get trajectory without update
    Notice: the sample process is ASYNC to maximize the utilization of IssacGym.
    :return: trajectory (env, step, :)
    """
    '''init'''
    if agent is None:
        agent = copy.deepcopy(args.agent)
        agent.save_or_load_agent(cwd=args.work_dir, if_save=False)

    if state_norm is not None:
        state_mean, state_var = state_norm.get()

    if (target_step is None) and (target_turn is None):
        target_step = args.update_step
        target_turn = args.update_turn

    '''main process'''
    s = envs.obs_dict
    running_flag = True

    traj = {'s': [], 'a': [], 'r': [], 'd': [], 'prob': []}
    traj_cnt = 0
    reward_sum = 0
    step = 0

    while running_flag:
        '''get action'''
        s = s['obs']
        if state_norm is None:
            _s = s
        else:
            _s = do_state_norm(s, state_mean, state_var)
        a_avg, a_std, a_noise, logprob = agent.select_actions(_s)
        if rand_act:
            a = a_noise
        else:
            a = a_avg
        action = a.clamp(-1, 1)  # clamp(-1, 1), tanh()

        '''env step'''
        s_, reward, done, _ = envs.step(action)

        traj['s'].append(s.cpu())
        traj['a'].append(a.cpu())
        traj['r'].append(reward.cpu())
        traj['d'].append(done.cpu())
        traj['prob'].append(logprob.cpu())

        reward_sum += torch.sum(reward).cpu().numpy()
        traj_cnt += torch.sum(done).cpu().numpy()
        step += done.shape[0]

        '''if end'''
        if ((target_step is None) or step >= target_step) and \
                ((target_turn is None) or traj_cnt >= target_turn):
            running_flag = False
        else:
            s = s_

    for key in ('s', 'a', 'r', 'd', 'prob'):
        traj[key] = torch.stack(traj[key], dim=0)

    return traj, step, reward_sum, traj_cnt


def explore(args, envs, agent=None, target_step=None, target_turn=None, state_norm=None, rand_act=True):
    """
    Get trajectory without update
    Notice: the sample process is 'ASYNC-SYNC' to balance sample efficiency & distribution
    a |-1- ---2--- -----3----- ------4| ------
    b |----1---- ----2---- -----3-----|
    c |--1-- --2-- --3-- --4-- --5-- -| -6--
    Suppose we've got enough sample with abc[123] after b[3] (sync checking).
    For stable of step size, I drop a[4]c[456]. For utilization maximization, finishing a[4]c[6] & collecting all is another option.
    :return: trajectory (step, :)
    """
    '''init'''
    if agent is None:
        agent = copy.deepcopy(args.agent)
        agent.save_or_load_agent(cwd=args.work_dir, if_save=False)

    if state_norm is not None:
        state_mean, state_var = state_norm.get()

    if (target_step is None) and (target_turn is None):
        target_step = args.update_step
        target_turn = args.update_turn

    env_maxstep = args.env_maxstep

    Keys = ('s', 'a', 'r', 'd', 'prob')
    '''main process'''
    for env in envs:
        env.reset()

    s = [None for _ in envs]
    traj_env = [[{'s': [], 'a': [], 'r': [], 'd': [], 'prob': []}] for _ in envs]
    traj_reward = [[0] for _ in envs]
    running_flag = True

    traj = {'s': [], 'a': [], 'r': [], 'd': [], 'prob': []}
    total_reward = 0
    traj_cnt = 0

    sync_pos = 1
    sync_cnt = len(envs)

    for i, env in enumerate(envs):
        s[i] = env.recv()

    while running_flag:
        '''get action'''
        if state_norm is None:
            _s = s
        else:
            _s = do_state_norm(np.array(s), state_mean, state_var)
        a_avg, a_std, a_noise, logprob = agent.select_actions(_s)
        if rand_act:
            # Reparameterized trick for Gaussian distribution
            a = a_noise
        else:
            a = a_avg
        action = a.clamp(-1, 1).numpy()  # clamp(-1, 1), tanh()

        '''env step'''
        for i, env in enumerate(envs):
            env.step(action[i])

        for i, env in enumerate(envs):
            s_, reward, done, _ = env.recv()
            if not running_flag:
                continue

            if (env_maxstep is not None) and (len(traj_env[i][-1]['r']) >= env_maxstep - 1):
                done = True

            traj_env[i][-1]['s'].append(s[i])
            traj_env[i][-1]['a'].append(a[i])
            traj_env[i][-1]['r'].append(reward)
            traj_env[i][-1]['d'].append(done)
            traj_env[i][-1]['prob'].append(logprob[i])
            traj_reward[i][-1] += reward

            if done:
                env.reset()
                if len(traj_env[i]) == sync_pos:
                    '''get it'''
                    total_reward += traj_reward[i][sync_pos - 1]
                    for key in Keys:
                        traj[key].extend(traj_env[i][sync_pos - 1][key])
                    traj_cnt += 1
                    '''sync check'''
                    sync_cnt -= 1
                    if sync_cnt == 0:
                        sync_pos += 1
                        sync_cnt = len(envs)
                        '''if end'''
                        if ((target_step is None) or len(traj['r']) >= target_step) and \
                                ((target_turn is None) or traj_cnt >= target_turn):
                            running_flag = False
                        '''scan'''
                        for j in range(len(envs)):
                            if len(traj_env[j]) > sync_pos:
                                '''get it'''
                                total_reward += traj_reward[j][sync_pos - 1]
                                for key in Keys:
                                    traj[key].extend(traj_env[j][sync_pos - 1][key])
                                traj_cnt += 1
                                sync_cnt -= 1

                traj_env[i].append({'s': [], 'a': [], 'r': [], 'd': [], 'prob': []})
                traj_reward[i].append(0)
                s[i] = env.recv()
            else:
                s[i] = s_

    for key in ('s', 'a', 'r', 'd', 'prob'):
        traj[key] = np.stack(traj[key], axis=0)

    return traj, len(traj['r']), total_reward, traj_cnt


def train(args, env=None, agent=None, buffer=None, target_step=None, start_step=0, state_norm=None):
    """
    :param args:
    :param env:
    :param agent:
    :param train_step:
    :param start_step:
    :return:
    """
    '''init'''
    if env is None:
        env = args.env

    if agent is None:
        agent = copy.deepcopy(args.agent)
        agent.save_or_load_agent(cwd=args.work_dir, if_save=False)

    if buffer is None:
        buffer = args.buffer

    if target_step is None:
        target_step = args.break_step

    if args.env_is_vec:
        explore_func = explore_issacgym
    else:
        explore_func = explore

    '''args'''
    work_dir = args.work_dir
    logger = args.logger
    batch_size = args.batch_size
    update_step = args.update_step
    repeat_times = args.repeat_times
    soft_update_tau = args.soft_update_tau

    step_cnt = start_step

    is_training = True
    while is_training:
        if (len(args.update_turn_inc) > 0) and (step_cnt > args.update_turn_inc[0]):
            args.update_turn *= 2
            args.update_turn_inc.pop(0)

        trajectory, step_, reward, traj_cnt = explore_func(args,
                                                           envs=env,
                                                           agent=agent,
                                                           target_step=update_step,
                                                           target_turn=args.update_turn,
                                                           state_norm=state_norm,
                                                           rand_act=True)
        buffer.reset()
        update_buffer(args, buffer, trajectory, key_dict=args.buffer_key, state_norm=state_norm)
        buffer.update_len()
        obj_critic, obj_actor = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        logger.add_scalar('traj-len_train', step_ / traj_cnt, global_step=step_cnt)
        logger.add_scalar('reward_train', reward / traj_cnt, global_step=step_cnt)
        logger.add_scalar('loss_critic', obj_critic, global_step=step_cnt)
        logger.add_scalar('loss_actor', obj_actor, global_step=step_cnt)
        logger.add_scalar('lr_critic', agent.cri_lr_scheduler.get_last_lr(), global_step=step_cnt)
        logger.add_scalar('action_std_shift', agent.act.a_std_log_shift.mean().exp(), global_step=step_cnt)
        step_cnt += step_
        if step_cnt >= target_step:
            is_training = False

    return env, agent, step_cnt, traj_cnt


def evaluate(args, cur_step, env=None, agent=None, state_norm=None):
    """
    :param args:
    :param env:
    :param agent:
    :return:
    """
    '''init'''
    if env is None:
        env = args.eval_env
    if env is None:
        env = args.env

    if agent is None:
        print('Warning: eval with newborn agent.')
        agent = copy.deepcopy(args.agent)
        agent.save_or_load_agent(cwd=args.work_dir, if_save=False)

    if args.env_is_vec:
        explore_func = explore_issacgym
    else:
        explore_func = explore

    '''args'''
    eval_turns = args.eval_turns

    trajectory, step_, reward, traj_cnt = explore_func(args,
                                                       envs=env,
                                                       agent=agent,
                                                       target_turn=eval_turns,
                                                       state_norm=state_norm,
                                                       rand_act=False)

    '''log'''
    args.logger.add_scalar('traj-len_eval', step_ / traj_cnt, global_step=cur_step)
    args.logger.add_scalar('reward_eval', reward / traj_cnt, global_step=cur_step)

    return env, agent, reward / traj_cnt


def run(args):
    start_time = time.time()
    print('-----------------------------------------------------------------')
    print(f'[{time_text(time.time() - start_time)}] Init process started.')

    envs = init_before_training(args)

    '''args'''
    work_dir = args.work_dir
    logger = args.logger
    break_step = args.break_step
    eval_step = args.eval_step

    '''init: Agent'''
    agent = args.agent
    # agent.save_or_load_agent(cwd=work_dir, if_save=False)

    '''init ReplayBuffer'''
    buffer = args.buffer

    if args.auto_state_norm:
        from .util import StateNorm
        state_norm = StateNorm(change_ratio_min=0)
    else:
        state_norm = None

    '''main process'''
    print(f'[{time_text(time.time() - start_time)}] Main process started.')
    print(f'SaveDir: {work_dir}')
    print('For details, view timely results in SaveDir via Tensorboard.\n'
          r'Example cmd: tensorboard --logdir=./log --bind_all')

    step = 0
    traj_cnt = 0
    while step < break_step:
        target_step = eval_step * ((step // eval_step) + 1)
        env, agent, step_, traj_cnt_ = train(args,
                                             env=envs,
                                             agent=agent,
                                             buffer=buffer,
                                             target_step=target_step,
                                             start_step=step,
                                             state_norm=state_norm)
        step = step_
        traj_cnt += traj_cnt_

        logger.add_scalar('traj_cnt', traj_cnt, global_step=step)
        agent.save_or_load_agent(work_dir, if_save=True)
        buffer.save_or_load_history(work_dir, if_save=True) if not agent.if_on_policy else None

        evaluate(args,
                 env=envs,
                 agent=agent,
                 cur_step=step,
                 state_norm=state_norm)

    if args.env_is_vec:
        pass
    else:
        for env in envs:
            env.close()
    print(f'[{time_text(time.time() - start_time)}] Process ended.')
    print('-----------------------------------------------------------------')
