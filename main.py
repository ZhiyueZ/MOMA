import random
import numpy as np
import scipy.stats as stats
import pickle
import argparse
import scipy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
def readParser():
    parser = argparse.ArgumentParser(description='random walk experiment')
    parser.add_argument('--seed', type=int, default=123, 
                        help='random seed (default: 123)')
    parser.add_argument('--timeout', type=int, default=1000, 
                        help='environment timeout')
    parser.add_argument('--mc_traj', type=int, default=500, 
                        help='Number of Monte Carlo samples')
    parser.add_argument('--max_iter', type=int, default=20, 
                        help='Number of iterations')
    parser.add_argument('--model_steps', type=int, default=150, 
                        help='Number of iterations for the model update')
    parser.add_argument('--actor_steps', type=int, default=150, 
                        help='Number of iterations for the actor update')                                           
    parser.add_argument('--lam_weight', type=float, default=3.0, 
                        help='adversarial parameter')
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='learning rate for actor')
    parser.add_argument('--eta', type=float, default=0.1, 
                        help='learning rate for model')                              
    parser.add_argument('--ridge_coeff', type=float, default=0, 
                        help='ridge coefficient')                              
    parser.add_argument('--gamma', type=float, default=0.4, 
                        help='discount factor')                                                                                   
    parser.add_argument('--data_path', type=str, default='simple',
                    help='data path')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='mini-batch size')        
    parser.add_argument('--feature_size', type=int, default=5, 
                        help='feature size')                         
    return parser.parse_args()

    

class policyClass:
    def __init__(self,feature_size,learning_rate=0.1):
        self.theta = np.zeros(3*feature_size)
        self.actions = [-1,0,1]
        self.alpha = learning_rate
        self.feature_size = feature_size
    def get_dist(self,state):
        inners = []
        for i in range(len(self.actions)):
            phi = phi_calculation(state,self.actions[i],feature_size=self.feature_size)
            inners.append(np.inner(self.theta,phi))
        pi = scipy.special.softmax(inners).tolist()
        return pi
    
    def get_action(self,state):
        u = np.random.uniform(0,1,size = None)
        pi = self.get_dist(state)
        if u<pi[0]:
            a = -1
        elif u>=pi[0] and u<pi[1]+pi[0]:
            a = 0
        else:
            a = 1
        return a
    
    def update_policy(self,target):
        self.theta = self.theta + self.alpha * target






def em_gmm_eins(xs, pis, mus, sigmas, tol=0.01, max_iter=100):

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j, (pi, mu, sigma) in enumerate(zip(pis, mus, sigmas)):
            ws[j, :] = pi * mvn(mu, sigma).pdf(xs)
        ws /= ws.sum(0)

        # M-step
        pis = np.einsum('kn->k', ws)/n
        mus = np.einsum('kn,np -> kp', ws, xs)/ws.sum(1)[:, None]
        sigmas = np.einsum('kn,knp,knq -> kpq', ws,
            xs-mus[:,None,:], xs-mus[:,None,:])/ws.sum(axis=1)[:,None,None]

        # update complete log likelihoood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi*mvn(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas



pos_error = 0.1
neg_error = 0.1
pos_end,neg_end = 3,-3
pos_step,neg_step = 2,-2


def phi_calculation(state,action,feature_size=5):
    phi = np.zeros(3*feature_size)
    if action == -1:
        for i in range(feature_size):
            phi[3*i] = np.exp(-0.5*pow(state-(4-2*i),2))
    elif action == 0:
        for i in range(feature_size):
            phi[3*i+1] = np.exp(-0.5*pow(state-(4-2*i),2))
    elif action == 1:
        for i in range(feature_size):
            phi[3*i+2] = np.exp(-0.5*pow(state-(4-2*i),2))
    return phi



def current_reward(s,sprime):
    if sprime >= pos_end:
        r = 0
    
    elif sprime <= neg_end:
        r = 0
    
    else:
        if sprime<0:
            r = -2
        else:
            r = -1.8
    return r

def termination(state):
    if state >= pos_end or state <= neg_end:
        done = True
    else:
        done = False
    return done
        


def get_next(psi,state,action):
    if action<0:
        error = neg_error
    else :
        error = pos_error 
    if action == -1:
        u = np.random.uniform(0,1,size = None)
        if u < psi[0]:
            ds = neg_step
        else:
            ds = 0
        mean = state + ds
    elif action == 0:
        u = np.random.uniform(0,1,size = None)
        if u < psi[1]:
            ds = 0
        else:
            ds = pos_step
        mean = state + ds
    else:
        u = np.random.uniform(0,1,size = None)
        if u < psi[2]:
            ds = 0
        else:
            ds = pos_step
        mean = state + ds
    sprime = np.random.normal(loc=mean, scale=error, size=None)
    return sprime

def sampler(policy, psi, gamma =0.4):
    s0 = np.random.uniform(low=-2, high=2, size=None)  # initial dist
    s = s0
    a = policy.get_action(s)
    done = False
    while not done:
        u = np.random.uniform(low=0.0, high=1.0, size=None)
        if u<gamma:
            next_s = get_next(psi,s,a)
            next_a = policy.get_action(next_s)
            done = termination(next_s)
            s = next_s
            a = next_a
        else:
            return [s,a]
    return [s,a]

def dynamics_MLE(dataset):
    states,actions,s_prime = dataset['observations'],dataset['actions'],dataset['next_observations']
    neg_inds,zero_inds,pos_inds = np.where(actions==-1)[0].tolist(),np.where(actions==0)[0].tolist(),np.where(actions==1)[0].tolist()
    del_s0,del_s1,del_s2 = np.take(s_prime,neg_inds)-np.take(states,neg_inds),np.take(s_prime,zero_inds)-np.take(states,zero_inds),np.take(s_prime,pos_inds)-np.take(states,pos_inds)
    #initialize EM 
    pis,mus,sigmas = np.random.random(2),np.random.random(2),np.random.random(2)
    _,pi0,mu0,_ = em_gmm_eins(del_s0.reshape(-1,1), pis, mus, sigmas)
    _,pi1,mu1,_ = em_gmm_eins(del_s1.reshape(-1,1), pis, mus, sigmas)
    _,pi2,mu2,_ = em_gmm_eins(del_s2.reshape(-1,1), pis, mus, sigmas)  

    ind0,ind1,ind2 = np.argmin(mu0), np.argmin(mu1),np.argmin(mu2)
    psi0,psi1,psi2 = pi0[ind0],pi1[ind1],pi2[ind2]
    psi_mle = np.array([psi0,psi1,psi2])
    return psi_mle


def grad_psi(psi,state,a,s_prime):
    grad = np.zeros(3,dtype=np.float32)
    ds = s_prime - state
    if a<0:
        error = neg_error
    else :
        error = pos_error 
    if a == -1:
        grad[0] = (norm.pdf(ds,neg_step,error)-norm.pdf(ds,0,error))/(psi[0]*norm.pdf(ds,neg_step,error)+(1-psi[0])*norm.pdf(ds,0,error))
    elif a == 0:
        grad[1] = (norm.pdf(ds,0,error)-norm.pdf(ds,pos_step,error))/(psi[1]*norm.pdf(ds,0,error)+(1-psi[1])*norm.pdf(ds,pos_step,error))
    else:
        grad[2] = (norm.pdf(ds,0,error)-norm.pdf(ds,pos_step,error))/(psi[2]*norm.pdf(ds,0,error)+(1-psi[2])*norm.pdf(ds,pos_step,error))
    return grad
def grad_psi_batch(psi,state,a,s_prime):
    grad =  np.zeros([3,len(a)],dtype=np.float32)
    for i in range(len(a)):
        if a[i]<0:
            error = neg_error
        else :
            error = pos_error 
        ds = s_prime[i] - state[i]
        if a[i] == -1:
            grad[0,i] = (norm.pdf(ds,neg_step,error)-norm.pdf(ds,0,error))/(psi[0]*norm.pdf(ds,neg_step,error)+(1-psi[0])*norm.pdf(ds,0,error))
        elif a[i] == 0:
            grad[1,i] = (norm.pdf(ds,0,error)-norm.pdf(ds,pos_step,error))/(psi[1]*norm.pdf(ds,0,error)+(1-psi[1])*norm.pdf(ds,pos_step,error))
        else:
            grad[2,i] = (norm.pdf(ds,0,error)-norm.pdf(ds,pos_step,error))/(psi[2]*norm.pdf(ds,0,error)+(1-psi[2])*norm.pdf(ds,pos_step,error))
    grad_out = np.mean(grad,axis=-1)
    return grad_out
def MC_V(s0, policy, psi, gamma =0.4, horizon = 10, num_traj = 500):
    dist = policy.get_dist(s0)
    V=[]
    for t in range(num_traj):
        s = s0
        reward = []
        done = False
        h = 0
        while not done:
            a = policy.get_action(s)
            dist = policy.get_dist(s)
            next_s = get_next(psi,s,a)
            r_sa = current_reward(s,next_s)
            reward.append(pow(gamma,h)*r_sa)
            s = next_s
            h += 1
            done = termination(s)
        V.append(np.sum(reward))
    return np.mean(V),np.std(V)

def MC_Q(s0, a0, policy, psi, gamma =0.4, horizon = 10, num_traj = 500):
    Q=[]
    for t in range(num_traj):
        s = s0
        a = a0
        reward = []
        done = False
        h = 0
        while not done:
            next_s = get_next(psi,s,a)
            r_sa = current_reward(s,next_s)
            reward.append(pow(gamma,h)*r_sa)
            s = next_s
            a = policy.get_action(s)
            done = termination(s)
        Q.append(np.sum(reward))
    return np.mean(Q)



def ridge_solver(target, features,coeff):
    A = np.identity(features.shape[1])
    A_biased = coeff * A
    if coeff==0:
        thetas = np.linalg.lstsq(features, target,rcond=None)[0]
    else:
        thetas = np.linalg.inv(features.T.dot(features) + A_biased).dot(features.T).dot(target)
    return thetas

    





def main(args=None):
    if args is None:
        args = readParser()
    # load dataset
    dataset_path = f'data/{args.data_path}.pkl'
    with open(dataset_path, 'rb') as f:
        dataset,trajectories = pickle.load(f)

    seednum = args.seed
    random.seed(seednum)
    np.random.seed(seednum)

    # MLE estimate
    psi_MLE = dynamics_MLE(dataset)
    psi_truth = np.array([0.6,0.6,0.4])
    print("*"*80)
    print(f'MLE: {psi_MLE}')
    print("*"*80)
    feature_size = args.feature_size
    #initialize the model params with MLE
    psi_current = psi_MLE
    alpha = args.alpha
    eta = args.eta
    gamma = args.gamma
    lam_weight = args.lam_weight
    lam_ridge = args.ridge_coeff
    MC_num = args.mc_traj
    offline_states,offline_actions,offline_sprime = dataset['observations'],dataset['actions'],dataset['next_observations']
    horizon = args.timeout
    policy = policyClass(feature_size,learning_rate=alpha)


    pi0_save = np.zeros([args.max_iter,3])
    psi_save = np.zeros([args.max_iter,3])
    std_save = []

    # main loop
    # for plotting
    value_vec0,value_vec1,value_vecm2 = [],[],[]
    for iter in range(args.max_iter):
        print(f'iteration: {iter}')
        obj_vec = []
        if args.model_steps >0:
            for j in range(args.model_steps):        
                sa_pair = sampler(policy=policy, psi=psi_current,gamma=gamma)
                state,action = sa_pair[0],sa_pair[1]
                next_state = get_next(psi_current,state,action)
                reward = current_reward(state,next_state)
                Value_s,temp = MC_V(state, policy, psi_current, gamma, horizon = horizon, num_traj = MC_num)
                advantage = reward + gamma * Value_s
                grad_sprime = grad_psi(psi_current,state,action,next_state)
                adv_objective = advantage * grad_sprime
                obj_vec.append(adv_objective)
            grad_T = grad_psi_batch(psi_current,offline_states,offline_actions,offline_sprime)
            model_update = np.mean(obj_vec,axis=0) - lam_weight * grad_T
            print(f"update: {model_update}, model grad: {np.mean(obj_vec,axis=0)}, MLE err: {grad_T}")
            psi_current -= eta * model_update

        print(f'psi:{psi_current}')
        
        
        X = []
        targets = []
        for k in range(args.actor_steps):
            #print(f"actor iter: {k}")
            sa_pair = sampler(policy, psi_current,gamma=gamma)
            action = np.random.randint(-1, high=2, size=None, dtype=int)
            value = MC_Q(s0=sa_pair[0],a0=action, policy=policy, psi=psi_current, gamma=gamma, horizon = horizon, num_traj = MC_num)
            targets.append(value)
            X.append(phi_calculation(sa_pair[0],action))
       
        X = np.array(X)
        targets = np.array(targets)
        w = ridge_solver(targets,X,lam_ridge)
        policy.update_policy(w)
        
        V_eval0,std_0 = MC_V(0.1, policy, psi=psi_truth, gamma=gamma, horizon = horizon, num_traj = MC_num)
        value_vec0.append(V_eval0) 
        V_eval1,temp = MC_V(2, policy, psi=psi_truth, gamma=gamma, horizon = horizon, num_traj = MC_num)
        value_vec1.append(V_eval1) 
        V_evalm2,temp = MC_V(-2, policy, psi=psi_truth, gamma=gamma, horizon = horizon, num_traj = MC_num)
        value_vecm2.append(V_evalm2) 
        policy_dist0 = policy.get_dist(0.1)
        policy_dist1 = policy.get_dist(2)
        policy_distm1 = policy.get_dist(-1)
        print(f'policy_0:{policy_dist0}')
        print(f'policy_2:{policy_dist1}')
        print(f'policy_-1:{policy_distm1}')
        print(f'value_0: {V_eval0}')
        print(f'value_2: {V_eval1}')
        print(f'value_-2: {V_evalm2}')
        pi0_save[iter] = policy_dist0
        std_save.append(std_0)
        psi_save[iter] = psi_current
    print("finished")        
    print(f'psi:{psi_current}')
    policy_dist = policy.get_dist(1)
    print(f'policy:{policy_dist}')
    
    # plot the value function V(0)
    fig, ax = plt.subplots()

    
    ax.plot(value_vec0)
    ax.set_title("v(0) vs iteration T")
   
    ax.set_xlabel("Iteration")
    ax.set_ylabel("V(0)")
    output_path = f"V_0.png"
    plt.savefig(output_path, dpi=200,bbox_inches='tight')

    fig, ax = plt.subplots()

    
    ax.plot(value_vec1)
    ax.set_title("v(1) vs iteration T")
   
    ax.set_xlabel("Iteration")
    ax.set_ylabel("V(1)")
    output_path = f"V_1.png"
    plt.savefig(output_path, dpi=200,bbox_inches='tight')

    weights = policy.theta
    with open("result.pkl", 'wb') as f:
        pickle.dump([value_vec0,pi0_save,weights,std_save,psi_save],f)




if __name__ == '__main__':
    main()