import math
import torch


class Arguments:
    def __init__(self):
        """Arguments"""
        '''Arguments for overall'''
        self.env_maxstep = 2000  # max step per trial for environment
        self.agent_load_dir = None

        self.state_dim = (200, 200)
        self.state1d_dim = 20
        self.action_dim = 3

        '''Arguments for device'''
        self.gpu_id = -1
        self.visible_gpu = f'{self.gpu_id}'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.device = torch.device(f"cuda:{self.gpu_id}" if (torch.cuda.is_available() and (self.gpu_id >= 0)) else "cpu")
        self.random_seed = None  # initialize random seed in self.init_before_training()

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1  # an approximate target reward usually be closed to 256
        self.auto_state_norm = False  # cumulate state statics through batches

        self.if_on_policy = True
        self.if_gae = True

        self.learning_rate = 3e-5
        self.learning_rate_critic = self.learning_rate
        self.learning_rate_policy = self.learning_rate * 1.
        self.net_dim = 256
        self.activation = torch.nn.ReLU  # ReLU, PReLU, GELU
        self.optimizer = torch.optim.Adam

        self.agent_kwargs = dict(if_per_or_gae=self.if_gae,
                                 ratio_clip=0.2,
                                 lambda_entropy=0.05,
                                 lambda_a_value=1.00,
                                 lambda_gae_adv=0.98,
                                 )
        self.lambda_a_value = 1.00
        self.lambda_gae_adv = 0.98

        self.batch_size = 4096
        # Dont decay the LR, increase the BS, ICLR 2018
        self.N_Bs = 2
        self.repeat_times = 2 * self.N_Bs
        self.soft_update_tau = 1 - math.pow(0.1, 1.0/self.N_Bs)  # smooth critic

        self.update_step = 1e3  # the minimum step between update
        self.update_start = max(self.batch_size * 2, 5e4)
        self.update_turn = 1  # the minimum turn between update
        self.update_turn_inc = []

        self.buffer_gpu = self.gpu_id
        self.buffer_maxlen = int(1e5)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
        self.lr_scheduler_kwargs = dict(milestones=[100 * self.repeat_times,
                                                    ],
                                        gamma=1.)

        '''Arguments for evaluate and save'''
        self.para_dir = 'deepnav/scripts/agent'
        self.logger = None  # SummaryWriter(log_dir=args.work_dir)

args = Arguments()
