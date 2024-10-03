
import random
import numpy as np
from tqdm import tqdm
from sim_env import SimEnv
import scipy.stats as stats
import pickle
import argparse
def readParser():
    parser = argparse.ArgumentParser(description='random walk experiment data generation')
    parser.add_argument('--seed', type=int, default=123, 
                        help='random seed (default: 123)')
    parser.add_argument('--timeout', type=int, default=1000, 
                        help='environment timeout')
    parser.add_argument('--save', type=str, default='simple',
                    help='save path')
    parser.add_argument('--num_traj', type=int, default=100, 
                        help='number of trajectories')                
 
    return parser.parse_args()





def behavioral_policy(state):
    weights = [0.9,0.05,0.05]
    actions = [-1,0,1]
    action = np.random.choice(actions,size=1,p=weights,replace=False).item()
    return action

def gen_data(env,args):
        state = env.reset()
        total_reward = 0
        actions = np.array([],dtype=np.int16)
        rewards = np.array([],dtype=np.float32)
        next_states = np.array([],dtype=np.float32)
        done = False
        dones = np.array([],dtype=bool)
        states = np.array([],dtype=np.float32)
        while (not done):
            action = behavioral_policy(state)
            # record data pre step
            actions = np.concatenate([actions,np.array([action])],axis=0)
            states = np.concatenate([states,np.array([state])],axis=0)
            # execute the action
            next_state, reward, done, _= env.step(action)
            #record data post step
            rewards = np.concatenate([rewards,np.array([reward])],axis=0)
            dones = np.concatenate([dones,[done]],axis=0)
            next_states = np.concatenate([next_states,np.array([next_state])],axis=0)
            total_reward += reward
            state = next_state
        
        traj_data = {}
        traj_data['observations'] = states
        traj_data['next_observations'] = next_states
        traj_data['actions'] = actions
        traj_data['terminals'] = dones
        traj_data['rewards'] = rewards

        return traj_data


def main(args=None):
    if args is None:
        args = readParser()
    seednum = args.seed
    random.seed(seednum)
    np.random.seed(seednum)
    save_path = 'data/'+args.save + '.pkl'
    env = SimEnv(timeout=args.timeout)
    dataset_traj=[]
    dataset = {}
    observations = np.array([],dtype=np.float32)
    actions = np.array([],dtype=np.int16)
    next_observations = np.array([],dtype=np.float32) 
    terminals = np.array([],dtype=bool)
    rewards = np.array([],dtype=np.float32)
    # populating dataset
    for i in tqdm(range(args.num_traj)):
        traj = gen_data(env,args)
        dataset_traj.append(traj)
        observations = np.concatenate([observations,traj['observations']],axis=0)
        actions = np.concatenate([actions,traj['actions']],axis=0)
        next_observations = np.concatenate([next_observations,traj['next_observations']],axis=0)
        terminals = np.concatenate([terminals,traj['terminals']],axis=0)
        rewards = np.concatenate([rewards,traj['rewards']],axis=0)

    dataset['observations'] = observations
    dataset['actions'] = actions
    dataset['next_observations'] = next_observations
    dataset['terminals'] = terminals
    dataset['rewards'] = rewards

    with open(save_path, 'wb') as f:
        pickle.dump([dataset,dataset_traj],f)
    






if __name__ == '__main__':
    main()