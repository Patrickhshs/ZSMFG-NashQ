import numpy as np
import itertools



class myQTable():
    
        def __init__(self,n_states_x=3 ,n_steps_state=5,history_table=None):
            self.n_states_x=n_states_x
            self.n_steps_state=n_steps_state # big N in the simplex discretization 
            self.n_steps_ctrl = 3
            self.history_table = history_table

        def init_states(self):
            
            combi_mu = itertools.product(np.linspace(0,self.n_steps_state,self.n_steps_state+1,dtype=int), repeat=self.n_states_x) #cartesian product; all possible distributions in the discretization of the simplex
            distributions_unnorm = np.asarray([el for el in combi_mu])
            states_tmp = distributions_unnorm.copy()
            self.states = states_tmp[np.where(np.sum(states_tmp, axis=1) == self.n_steps_state)] / float(self.n_steps_state)#shape:(5456,4)
            #print(self.states[1:3])
            self.n_states = np.shape(self.states)[0]
            #print(self.states)
             # as we have 

        def init_ctrl(self):
            combi_ctrl = itertools.product(np.linspace(0,self.n_steps_ctrl,self.n_steps_ctrl+1,dtype=int), repeat=3)# n_states_x) # cartesian product; all possible controls as functions of state_x
            controls = np.asarray([el for el in combi_ctrl]) # np.linspace(0,1,n_steps_ctrl+1)
            self.controls = controls[np.where(np.sum(controls, axis=1) == self.n_steps_ctrl)] / float(self.n_steps_ctrl)
            # combi_ctrl = itertools.product(np.linspace(0,1,self.n_steps_ctrl+1), repeat=self.n_states_x)#n_states_x) #cartesian product; all possible controls as functions of state_x
            # self.controls = np.asarray([el for el in combi_ctrl])
            self.n_controls = np.shape(self.controls)[0]
            #print("controls = {}".format(self.controls))
            print('MDP: n states = {}\nn controls = {}'.format(self.n_states, self.n_controls))
            if self.history_table is not None:
                self.Q_table = self.history_table
                
            else:

                self.Q_table = np.random.random((self.n_states, self.n_controls ,self.n_controls)) # shape:(state,action_1,action_2)
                #self.Q_table = np.ones((self.n_states, self.n_controls ,self.n_controls)) # shape:(state,action_1,action_2)
                #self.Q_table = np.zeros((self.n_states, self.n_controls ,self.n_controls)) # shape:(state,action_1,action_2)
            
            
            # Q_old[:,11] = 0.01
            # Q_new = np.zeros((n_states, n_controls))v 
            print("Q shape = {}".format(np.shape(self.Q_table)))
        
        def proj_W_index(self,mu):
            minimal = 999999
            index = 0
            for i in range(np.shape(self.states)[0]):
                distance = np.sum(np.abs(mu - self.states[i]))
                if distance < minimal:
                    minimal = distance
                    index = i
            #return np.argmin(map(lambda mu2: np.sum(np.abs(mu - mu2)), self.states))
            return index

        def get_state_index(self,state):

            for i in range(self.n_states):
                if np.array_equal(self.states[i] ,state):
                    return i