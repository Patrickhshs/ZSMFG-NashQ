import numpy as np
import argparse
import pprint as pp
import time











# ===========================
#   Environment for KFP equation
# ===========================

class MyEnvKFPCyberSecurity(object):


    def __init__(self):

        self.T = 0.2#0.5 # for one time step
        self.Nt = 1#5
        self.Dt = self.T / self.Nt # time step for KFP PDE
        self.NS = 4 # number of states
        self.beta_UU = 0.3 #infection rate for undefended to undefended computers
        self.beta_UD = 0.4 #infection rate for undefended to defended computers
        self.beta_DU = 0.3 #infection rate for defended to undefended computers
        self.beta_DD = 0.4 #infection rate for defended to defended computers
        self.v_H = 0.6 # attack intensity
        self.lambda_speed = 0.8 # speed of response
        self.q_rec_D = 0.5 #recover rate if computer is defended
        self.q_rec_U = 0.4 #recover rate if computer is undefended
        self.q_inf_D = 0.4 #infection rate if computer is defended
        self.q_inf_U = 0.3 #infection rate if computer is undefended
        self.k_D = 0.3 # cost of being defended
        self.k_I = 0.5 # cost of being infected
        #self.stateInit = [0.25, 0.25, 0.25, 0.25] # initial distribution
        self.stateInit = [0.01, 0.01, 0.48, 0.5]
        self.state = self.stateInit
        # self.running_cost_vec = np.zeros(self.NS)
        # for iS in range(self.NS):
        #     self.running_cost_vec[iS] = self.running_cost_t(iS)
        action_set = [0, 1] # actions
        action_space = []
        for iS in range(self.NS):# get the action space for each state
            action_space.append(action_set)

        # FOR DDPG
        self.state_dim = 4 #observation_space.shape[0]
        self.action_dim = 4 #env.action_space.shape[0]
        self.action_bound = 1.0 #env.action_space.high
        # no env.action_space.high and env.action_space.low, just use self.action_bound

        self.dt = self.Dt
        return

    # ==================== USEFUL FUNCTIONS
    def get_index(self, str):
        if str=='DI':
            return 0
        if str=='DS':
            return 1
        if str=='UI':
            return 2
        if str=='US':
            return 3

    # Version with a continuous alpha, truncated between 0 and 1
    def get_lambda_t_continuousAlpha(self, mu_t, alpha_t):#Get the transition matrix
        
        # see page 656 of volume I
        lambda_matrix = np.zeros((self.NS, self.NS))#Create empty matrtix
        lambda_matrix[self.get_index('DI'),self.get_index('DS')] = self.q_rec_D
        lambda_matrix[self.get_index('DS'),self.get_index('DI')] = self.v_H*self.q_inf_D + self.beta_DD*mu_t[self.get_index('DI')] + self.beta_UD*mu_t[self.get_index('UI')]
        lambda_matrix[self.get_index('UI'),self.get_index('US')] = self.q_rec_U
        lambda_matrix[self.get_index('US'),self.get_index('UI')] = self.v_H*self.q_inf_U + self.beta_UU*mu_t[self.get_index('UI')] + self.beta_DU*mu_t[self.get_index('DI')]
        alpha_t_trunc = min(1.0, max(0.0, alpha_t))#Select the alpha_t
        #alpha_t_trunc = alpha_t # already bounded in actor's definition ## BUT THERE IS NOISE FOR EXPLORATION
        #if alpha_t == 1:

        
        lambda_matrix[self.get_index('DI'),self.get_index('UI')] = alpha_t_trunc * self.lambda_speed
        lambda_matrix[self.get_index('DS'),self.get_index('US')] = alpha_t_trunc * self.lambda_speed
        lambda_matrix[self.get_index('UI'),self.get_index('DI')] = alpha_t_trunc * self.lambda_speed
        lambda_matrix[self.get_index('US'),self.get_index('DS')] = alpha_t_trunc * self.lambda_speed

        for iS in range(0,self.NS):#Fill the diagonal with negative sum of the entities in each row
            lambda_matrix[iS, iS] = - np.sum(lambda_matrix[iS])
        return lambda_matrix

    # Derivative of matrixlambda wrt measure
    def get_Dmu_lambda_t(self, iSderiv, mu_t, u_t, alpha_t):
        # see page 656 of volume I
        Dmu_lambda_matrix = np.zeros((self.NS, self.NS))
        if iSderiv == self.get_index('DI'):
            Dmu_lambda_matrix[self.get_index('DS'),self.get_index('DI')] = self.beta_DD#*mu_t[self.get_index('DI')]
            Dmu_lambda_matrix[self.get_index('US'),self.get_index('UI')] = self.beta_DU#*mu_t[self.get_index('DI')]
            Dmu_lambda_matrix[self.get_index('DS'),self.get_index('DS')] = -self.beta_DD#*mu_t[self.get_index('DI')]
            Dmu_lambda_matrix[self.get_index('US'),self.get_index('US')] = -self.beta_DU#*mu_t[self.get_index('DI')]
        if iSderiv == self.get_index('UI'):
            Dmu_lambda_matrix[self.get_index('DS'),self.get_index('DI')] = self.beta_UD#*mu_t[self.get_index('UI')]
            Dmu_lambda_matrix[self.get_index('US'),self.get_index('UI')] = self.beta_UU#*mu_t[self.get_index('UI')]
            Dmu_lambda_matrix[self.get_index('DS'),self.get_index('DS')] = -self.beta_UD#*mu_t[self.get_index('UI')]
            Dmu_lambda_matrix[self.get_index('US'),self.get_index('US')] = -self.beta_UU#*mu_t[self.get_index('UI')]
        return Dmu_lambda_matrix

    def running_cost_t(self, iS, mu):
        # running cost for given state and control
        rcost = 0
        #cost of the central planner if computers are defended
        if iS == self.get_index('DI') or iS == self.get_index('DS'):
            rcost += self.k_D

        #cost of the central planner if computers are undefended
        if iS == self.get_index('DI') or iS == self.get_index('UI'):
            rcost += self.k_I
        return rcost

    # def final_cost(iS):
    #     return 0

    def get_Hamiltonian(self, iS, mu_t, u_t, alpha_t):

        #Hamiltonian stands for the cost
        return np.matmul(self.get_lambda_t_continuousAlpha(mu_t, u_t, alpha_t)[iS], u_t) + self.running_cost_t(iS)

    # Derivatives of the Hamiltonian wrt measure at point iSderiv
    def get_Dmu_Hamiltonian(self, iS, iSderiv, mu_t, u_t, alpha_t):
        return np.matmul(self.get_Dmu_lambda_t(iSderiv, mu_t, u_t, alpha_t)[iS], u_t)

    def get_alphahat_t(self, iS, mu_t, u_t):
        # optimal control for given state, mu and u
        H_0 = self.get_Hamiltonian(iS, mu_t, u_t, 0)
        H_1 = self.get_Hamiltonian(iS, mu_t, u_t, 1)
        if H_0 <= H_1:
            return 0
        else:
            return 1

    def get_alphahat_t_vec(self, mu_t, u_t):
        alphahat = np.zeros(self.NS)
        for iS in range(self.NS):
            alphahat[iS] = self.get_alphahat_t(iS, mu_t, u_t)
        return alphahat

    def get_q_t_withActions(self, mu_t, alpha):
        # see equation (7.37)
        q_t = np.zeros((self.NS, self.NS))
        # alphahat = get_alphahat_t_vec(mu_t, u_t)

        # get the q table for different alpha
        for iS in range(self.NS):
            # q_t[iS] = self.get_lambda_t(mu_t, alpha[iS])[iS]# alphahat[iS])[iS]
            q_t[iS] = self.get_lambda_t_continuousAlpha(mu_t, alpha[iS])[iS]# alphahat[iS])[iS]
        return q_t

    # ==================== SOLVE PDE starting from mu with action profile alpha
    def get_mu_and_reward(self, mu, alpha):
        social_reward = 0 # reward for the social planner
        new_mu_prev = np.zeros(self.NS)
        new_mu = np.zeros(self.NS)
        new_mu_prev = mu # new state distribution; shape(4,1)

        # number of iteration
        for it in range(0,self.Nt):
            q_t = self.get_q_t_withActions(new_mu_prev, alpha)
            new_mu = np.matmul(new_mu_prev, np.eye(self.NS) + self.Dt*q_t)
            # self.running_cost_vec = np.zeros(self.NS)

            for iS in range(self.NS):#iterate each state
                # self.running_cost_vec[iS] = self.running_cost_t(iS, new_mu_prev)
                # social_reward += -self.running_cost_t(iS, new_mu_prev) * new_mu_prev[iS] # new_mu[iS]#
                social_reward += -np.inner(self.running_cost_t(iS, new_mu), new_mu[iS])#
                # self.running_cost_vec[iS] = self.running_cost_t(iS, new_mu_prev)
            # social_reward += np.inner(new_mu, -self.running_cost_vec)*self.Dt
            social_reward *= self.Dt
            new_mu_prev = new_mu.copy()
            #social_reward += np.inner(new_mu, self.running_cost_vec)*self.Dt
        return new_mu, social_reward


    def seed(self, seed):
        #self.seed = seed
        #np.random.seed(self.seed)
        return

    def reset(self):
        self.state = np.random.uniform(0,1,size=self.NS)
        self.state /= np.sum(self.state) # normalization
        return self.state

    def set_state(self, s):
        self.state = s
        return

    def step(self, action_as_array):
        action = action_as_array#self.actionArray[action_index]
        mu = self.state
        new_mu, social_reward = self.get_mu_and_reward(mu, action)
        self.state = new_mu
        terminal = False
        info = None
        return self.state, social_reward, terminal, info
