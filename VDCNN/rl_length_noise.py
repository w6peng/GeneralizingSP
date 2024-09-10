"""

This code implements the TopBits Selective Protection Technique

"""
# General imports
import os
import argparse
import math
from copy import deepcopy
import numpy as np

# Torch imports
import torch
import torch.backends.cudnn as cudnn

# Internal imports
from lib.utils.utils import prYellow, best_policy_list, export_results
from lib.env import ImportantBitsFloatFirstNEnv, ImportantBitsQuanFirstNEnv
from lib.ddpg import DDPG

#VDCNN imports
from lib.net import VDCNN
from lib.datasets import load_datasets
from torch.utils.data import DataLoader, Dataset
import lmdb

# Models
model_names = ['VDCNN']
print('support models: ', model_names)


def train(num_episode, agent, env, output, args, debug=False):
    '''
    Training function for finding best reward (weight mask), policy, and training cycle via Deep Reinforcement Learning using 
    specified agent and environment.
    
    Parameters:
        num_episode (int) :
            number of cycles to test for
        agent (DDPG) : 
            DDPG object containing actor/critic DRL networks & necessary DRL functions
        env (ImportantBitsEnv) :
            environment object for performing DRL actions with agent
        output (str) : 
            folder for saving model to at regular checkpoints
        args (ArgumentParser) :
            arguments specified at runtime. function-specific args include args.warmup and args.n_update.
        debug=False (bool) :
            specifies whether or not cycle info should be directly printed in addition to being written to an output file
    '''
    # best record
    best_reward = -math.inf
    best_policy = []
    best_episode_num = 0
    acc = 0.0
    ratio = 0.0
    best_history = []
    
    metric_dict = {'representation': args.representation,
                   'code': args.code,
                   'dataset': args.dataset, 
                   'preserve_ratio': args.preserve_ratio, 
                   'num_episodes': args.train_episode,
                   'load_agent_weights': bool(args.load_agent_weights)}

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
            if episode > args.warmup:
                # decay
                agent.step()

        # agent pick action ...
        # Random action returns value from uniform range [0,1] if warming up
        # Otherwise has agent select action based off of observation
        
        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        # Uses action to set policy, and adds to record
        # Note: reward changes based on episode (decay factor), which is why diff reward comes from same policy. Also has diff acc because diff noise added each time. 
        # Believe that it's just generally testing how it holds up to diff noise and gives reward based on robustness
        observation2, reward, done, info = env.step(action) #observation2: next layer embedding
        observation2 = deepcopy(observation2)
        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermediate model
        if episode % int(num_episode / 10) == 0:
            agent.save_model(output, args.dataset)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        debug = True
        
        # info['w_size'] is calculated as the current number of bits protected, with a multiplier of 1/8e6 converting from bits to megabytes
        
        if done:  # end of episode
            if debug:
                print('#{}: episode_reward: {:.4f} acc: {:.4f}, weight: {} MB'.format(episode, episode_reward,
                                                                                         info['accuracy'],
                                                                                         info['w_size'] * 1. / 8e6))
            text_writer.write(
                '#{}: episode_reward: {:.4f} acc: {:.4f}, weight: {} MB\n'.format(episode, episode_reward,
                                                                                     info['accuracy'],
                                                                                     info['w_size'] * 1. / 8e6))
            final_reward = T[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # Reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            # Set best reward info
            if final_reward > best_reward:
                metric_dict['ratio'] = info['w_ratio']
                metric_dict['acc'] = info['accuracy']
                metric_dict['best_policy'] = env.strategy
                metric_dict['best_reward'] = final_reward
                metric_dict['best_episode_num'] = episode
                
                best_history.append(['episode: {}'.format(metric_dict['best_episode_num']), 'reward: {:.4f}'.format(metric_dict['best_reward']), 'acc: {:.4f}'.format(metric_dict['acc']), 'ratio: {:.4f}'.format(metric_dict['ratio'])])

            text_writer.write('best_reward: {}\n'.format(metric_dict['best_reward']))
            text_writer.write('best_policy: {}\n'.format(metric_dict['best_policy']))
            
    best_list = best_policy_list(metric_dict['best_policy'], 'topbits', args.bit)
    text_writer.write('\nScheme: bitmask_{}_{}\n'.format(args.representation, args.code))
    text_writer.write('best_episode_num: {}\n'.format(metric_dict['best_episode_num']))
    text_writer.write('best_accuracy: {}\n'.format(metric_dict['acc']))
    text_writer.write('best_reward: {}\n'.format(metric_dict['best_reward']))
    text_writer.write('best_policy: {}\n'.format(metric_dict['best_policy']))
    text_writer.write('best_policy (list): {}\n'.format(best_list))
    text_writer.write('best_ratio: {}\n'.format(metric_dict['ratio']))
    if args.history:
        text_writer.write('best_outcome_history: {}\n'.format(best_history))
    text_writer.close()
    
    # Add additional performance metrics to dictionary to return
    metric_dict['best_policy'] = best_list
    #metric_dict['best_history'] = best_history
    
    return metric_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')

    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--dataset', default='ag_news', type=str, help='dataset to use)', choices=['ag_news', 'sogou_news', 'yelp_polarity'])
    parser.add_argument('--dataset_root', default=None, type=str, help='path to dataset)')
    parser.add_argument('--preserve_ratio', default=0.25, type=float, help='preserve ratio of the model size')
    parser.add_argument('--min_bit', default=0, type=int, help='minimum bit to use')
    parser.add_argument('--max_bit', default=32, type=int, help='maximum bit to use')
    parser.add_argument('--n_actions', default=10, type=int, help='size of actions')
    parser.add_argument('--bit', default=32, type=int, help='bitwidth of model to use')
    parser.add_argument('--float_bit', default=32, type=int, help='the bit of full precision float')
    parser.add_argument('--is_pruned', dest='is_pruned', action='store_true')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for critic')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=30, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=128, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--save_agent_weights', dest='save_agent_weights', action='store_true')
    parser.add_argument('--load_agent_weights', default=None, type=str, help='dataset to load weights for')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.995, type=float,
                        help='delta decay during exploration')
    parser.add_argument('--n_update', default=1, type=int, help='number of rl to update each time')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default=os.path.join('..', 'checkpoints', 'checkpoint_length'), type=str, help='output dir for drl weights and log file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=5000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=17, type=int, help='')
    # n_worker decreased to alleviate runtime crashes on nautilus cluster
    #parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')
    parser.add_argument('--n_worker', default=0, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=128, type=int, help='number of data batch size')
    parser.add_argument('--finetune_lr', default=0.001, type=float, help='finetune gamma')
    parser.add_argument('--finetune_epoch', default=1, type=int, help='')
    parser.add_argument('--use_top5', default=False, type=bool, help='whether to use top5 acc in reward')
    parser.add_argument('--train_size', default=20000, type=int, help='number of train data size')
    parser.add_argument('--val_size', default=30000, type=int, help='number of val data size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--representation', default='float', type=str, help='decide between floating point or fixed point', choices={'fixed', 'float'})
    parser.add_argument('--code', default='ideal', type=str, help='select between ideal code or BCH code', choices={'bch', 'ideal'})
    parser.add_argument('--history', default=True, type=bool, help='whether or not to display output of all best rewards/policy/etc')
    parser.add_argument('--out', dest='out', action='store_true', help='flag for outputting results (as opposed to just saving to file)')
    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='VDCNN', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: VDCNN)')
    # device options
    parser.add_argument('--gpu_id', default='7', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    
    assert 0 <= args.preserve_ratio <= 1, 'Preserve Ratio parameter must be bounded between 0, 1'

    
    if args.dataset_root is None:
        args.dataset_root = os.path.join('datasets', args.dataset, args.arch.lower(), 'test.lmdb')
    
    base_folder_name = '{}_{}'.format(args.arch, args.dataset)
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix
    args.output = os.path.join(args.output, base_folder_name)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    # tfwriter = SummaryWriter(logdir=args.output)
    text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
    print('==> Output path: {}...'.format(args.output))

    # Use CUDA
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    assert torch.cuda.is_available(), 'CUDA is needed for CNN'

    # Set seed so that actions taken will be reproducible 
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Create model
    if args.dataset == 'ag_news':
        n_classes = 4
    elif args.dataset == 'sogou_news':
        n_classes = 5
    elif args.dataset == 'yelp_polarity':
        n_classes = 2
        
    model = VDCNN(n_classes=n_classes, num_embedding=69, embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=True)
    state_dict = torch.load(os.path.join('results', 'models', 'vdcnn', args.dataset, 'model_epoch_100')) 
    
    model.load_state_dict(state_dict)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    cudnn.benchmark = True
    

    # Set code type
    if args.code == 'bch':
        if args.representation == 'fixed':
            code = (8191, 6787, 110) #fixed point
        if args.representation == 'float':
            code = (8191, 6722, 115) #floating point
    elif args.code == 'ideal':
        code = None

    # Create environment based off of representation type 
    if args.representation == 'fixed':
        env = ImportantBitsQuanFirstNEnv(model, args.dataset, args.dataset_root,
                      compress_ratio=args.preserve_ratio, num_workers=args.n_worker,
                      batch_size=args.data_bsize, args=args, bitwidth=args.bit, is_model_pruned=args.is_pruned, code=code)
    elif args.representation == 'float':
        env = ImportantBitsFloatFirstNEnv(model, args.dataset, args.dataset_root,
                      compress_ratio=args.preserve_ratio, num_workers=args.n_worker,
                      batch_size=args.data_bsize, args=args, bitwidth=args.bit, is_model_pruned=args.is_pruned, code=code)

    # Create agent + requirements and train in environment
    nb_states = env.layer_embedding.shape[1] # number of dimensions (layer type, #in, #out, stride, kernel size, weight size, input feature map size, layer index, max bit)
    nb_actions = 1 # actions for weight and activation quantization
    args.rmsize = args.rmsize * len(env.layer_idx)  # replay memory * num layers to evaluate
    print('** Actual replay buffer size: {}'.format(args.rmsize))
    agent = DDPG(nb_states, nb_actions, len(env.layer_idx), args)
    if args.load_agent_weights:
        agent.load_weights(folder='weights', dataset=args.load_agent_weights)

    # Train model
    metric_dict = train(args.train_episode, agent, env, args.output, args, debug=args.debug)   
    
    # Save model weights (if specified)
    if args.save_agent_weights:
        agent.save_model(folder='weights', dataset=args.dataset)
    
    # Output best policy information
    if args.out:
        print('Scheme: bitmask_{}_{}'.format(args.representation, args.code))
        print('best_episode_num: ', metric_dict['best_episode_num'])
        print('best_reward: ', metric_dict['best_reward'])
        print('best_policy: ', metric_dict['best_policy'])
        print('best_accuracy: ', metric_dict['acc'])
        print('best_ratio: ', metric_dict['ratio'])
    
    #if args.history:
    #    print('best_outcome_history: {}'.format(best_history))
    
    # Save best policy infoformation
    export_results(filename='results/drl/topbits.csv', data=metric_dict)