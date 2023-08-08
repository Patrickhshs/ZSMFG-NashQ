import numpy as np
import matplotlib.pyplot as plt



# ==================== PARAMETERS
T = 20.0
Nt = 2000 # number of time intervals
Dt = T / Nt
NS = 4 # number of states
beta_UU = 0.3
beta_UD = 0.4
beta_DU = 0.3
beta_DD = 0.4
v_H = 0.6 # attack intensity
lambda_speed = 0.8 # speed of response
q_rec_D = 0.5
q_rec_U = 0.4
q_inf_D = 0.4
q_inf_U = 0.3
k_D = 0.3 # cost of being defended
k_I = 0.5 # cost of being infected
#
discount_gamma = 0.5
discount_beta = - np.log(discount_gamma)


# ==================== USEFUL FUNCTIONS
def get_index(str):
    if str=='DI':
        return 0
    if str=='DS':
        return 1
    if str=='UI':
        return 2
    if str=='US':
        return 3

def get_lambda_t(mu_t, u_t, alpha_t):
    # see page 656 of volume I
    lambda_matrix = np.zeros((NS, NS))
    lambda_matrix[get_index('DI'),get_index('DS')] = q_rec_D
    lambda_matrix[get_index('DS'),get_index('DI')] = v_H*q_inf_D + beta_DD*mu_t[get_index('DI')] + beta_UD*mu_t[get_index('UI')]
    lambda_matrix[get_index('UI'),get_index('US')] = q_rec_U
    lambda_matrix[get_index('US'),get_index('UI')] = v_H*q_inf_U + beta_UU*mu_t[get_index('UI')] + beta_DU*mu_t[get_index('DI')]
    if alpha_t == 1:
        lambda_matrix[get_index('DI'),get_index('UI')] = lambda_speed
        lambda_matrix[get_index('DS'),get_index('US')] = lambda_speed
        lambda_matrix[get_index('UI'),get_index('DI')] = lambda_speed
        lambda_matrix[get_index('US'),get_index('DS')] = lambda_speed
    for iS in range(0,NS):
        lambda_matrix[iS, iS] = - np.sum(lambda_matrix[iS])
    return lambda_matrix

# Derivative of matrixlambda wrt measure
def get_Dmu_lambda_t(iSderiv, mu_t, u_t, alpha_t):
    # see page 656 of volume I
    Dmu_lambda_matrix = np.zeros((NS, NS))
    if iSderiv == get_index('DI'):
        Dmu_lambda_matrix[get_index('DS'),get_index('DI')] = beta_DD#*mu_t[get_index('DI')]
        Dmu_lambda_matrix[get_index('US'),get_index('UI')] = beta_DU#*mu_t[get_index('DI')]
        Dmu_lambda_matrix[get_index('DS'),get_index('DS')] = -beta_DD#*mu_t[get_index('DI')]
        Dmu_lambda_matrix[get_index('US'),get_index('US')] = -beta_DU#*mu_t[get_index('DI')]
    if iSderiv == get_index('UI'):
        Dmu_lambda_matrix[get_index('DS'),get_index('DI')] = beta_UD#*mu_t[get_index('UI')]
        Dmu_lambda_matrix[get_index('US'),get_index('UI')] = beta_UU#*mu_t[get_index('UI')]
        Dmu_lambda_matrix[get_index('DS'),get_index('DS')] = -beta_UD#*mu_t[get_index('UI')]
        Dmu_lambda_matrix[get_index('US'),get_index('US')] = -beta_UU#*mu_t[get_index('UI')]
    return Dmu_lambda_matrix


def running_cost_t(iS):
    # running cost for given state and control
    rcost = 0
    if iS == get_index('DI') or iS == get_index('DS'):
        rcost += k_D
    if iS == get_index('DI') or iS == get_index('UI'):
        rcost += k_I
    return rcost

def final_cost(iS):
    return 0

def total_cost(mu, it):
    c = 0
    disc_tmp = 1.0
    for its in range(it,Nt):
        c += disc_tmp * np.sum([running_cost_t(iS) for iS in range(NS)] * mu[its])*Dt
        disc_tmp *= np.exp(-discount_beta*Dt)
    c += disc_tmp * np.sum([final_cost(iS) for iS in range(NS)] * mu[Nt])
    return c

def total_cost_vec(mu):
    return [total_cost(mu, it) for it in range(0,Nt+1)]

def get_Hamiltonian(iS, mu_t, u_t, alpha_t):
    return np.matmul(get_lambda_t(mu_t, u_t, alpha_t)[iS], u_t) + running_cost_t(iS)

# Derivatives of the Hamiltonian wrt measure at point iSderiv
def get_Dmu_Hamiltonian(iS, iSderiv, mu_t, u_t, alpha_t):
    return np.matmul(get_Dmu_lambda_t(iSderiv, mu_t, u_t, alpha_t)[iS], u_t)

def get_alphahat_t(iS, mu_t, u_t):
    # optimal control for given state, mu and u
    H_0 = get_Hamiltonian(iS, mu_t, u_t, 0)
    H_1 = get_Hamiltonian(iS, mu_t, u_t, 1)
    if H_0 <= H_1:
        return 0
    else:
        return 1

def get_alphahat_t_vec(mu_t, u_t):
    alphahat = np.zeros(NS)
    for iS in range(NS):
        alphahat[iS] = get_alphahat_t(iS, mu_t, u_t)
    return alphahat

def get_q_t(mu_t, u_t):
    # see equation (7.37)
    q_t = np.zeros((NS, NS))
    alphahat = get_alphahat_t_vec(mu_t, u_t)
    for iS in range(NS):
        q_t[iS] = get_lambda_t(mu_t, u_t, alphahat[iS])[iS]
    return q_t



# ==================== SOLVE PDES
def solve_KFP(mu, u):
    new_mu = np.zeros((Nt+1, NS))
    new_mu[0] = mu[0]
    for it in range(0,Nt):
        q_t = get_q_t(mu[it], u[it])
        #print 'q_t = ', q_t
        new_mu[it+1] = np.matmul(new_mu[it], np.eye(NS) + Dt*q_t)
        #print 'it = ', it+1, 'new_mu = ', new_mu[it+1]
    return new_mu

def solve_HJB(mu, u):
    new_u = np.zeros((Nt+1, NS))
    for iS in range(NS):
        new_u[Nt][iS] = final_cost(iS)
    for it in range(Nt-1, -1, -1):
        opt_H = np.zeros(NS)
        Dmu_opt_H = np.zeros((NS, NS))
        for iS in range(NS):
            H0 = get_Hamiltonian(iS, mu[it+1], new_u[it+1], 0)
            H1 = get_Hamiltonian(iS, mu[it+1], new_u[it+1], 1)
            if H0 <= H1:
                #opt_H[iS] = min(get_Hamiltonian(iS, mu[it+1], new_u[it+1], 0), get_Hamiltonian(iS, mu[it+1], new_u[it+1], 1))
                opt_H[iS] = H0
                for iSderiv in range(NS):
                    Dmu_opt_H[iS][iSderiv] = get_Dmu_Hamiltonian(iS, iSderiv, mu[it+1], new_u[it+1], 0)
            else:
                opt_H[iS] = H1
                for iSderiv in range(NS):
                    Dmu_opt_H[iS][iSderiv] = get_Dmu_Hamiltonian(iS, iSderiv, mu[it+1], new_u[it+1], 1)

        #new_u[it] = new_u[it+1] - Dt*opt_H
        #new_u[it] = new_u[it+1] + Dt*opt_H + Dt*np.matmul(Dmu_opt_H, mu[it+1])
        new_u[it] = new_u[it+1] + Dt*opt_H + Dt*np.matmul(Dmu_opt_H, mu[it+1]) \
                        - Dt*discount_beta * new_u[it+1]
    return new_u

def solve_HJBKFP_Picard(mu_init, u_init, tol):
    u = u_init
    mu = mu_init
    new_mu = solve_KFP(mu, u)
    new_u = solve_HJB(new_mu, u)
    #print new_mu
    iiter = 1
    print 'iiter = ', iiter, '\t tol = ', tol, '\t np.norm(new_mu - mu) = ', np.linalg.norm(new_mu - mu), '\t np.norm(new_u - u) = ', np.linalg.norm(new_u - u)
    while np.linalg.norm(new_mu - mu)>tol or np.linalg.norm(new_u - u)>tol:
        u = new_u
        mu = new_mu
        new_mu = solve_KFP(mu, u)
        new_u = solve_HJB(new_mu, u)
        iiter +=1
        print 'iiter = ', iiter, '\t tol = ', tol, '\t np.norm(new_mu - mu) = ', np.linalg.norm(new_mu - mu), '\t np.norm(new_u - u) = ', np.linalg.norm(new_u - u)
    return new_mu, new_u

tol = 1.e-5
# itest = 3
for itest in range(1,1+3):
    if itest == 1:
        mu_0 = [0.25, 0.25, 0.25, 0.25]
        basename = 'test1'
    elif itest == 2:
        mu_0 = [1.0, 0., 0., 0.]
        basename = 'test2'
    elif itest == 3:
        mu_0 = [.0, 0., 0., 1.0]
        basename = 'test3'
    u_T = [0, 0, 0, 0]
    mu_init = np.zeros((Nt+1, NS))
    u_init = np.zeros((Nt+1, NS))
    for it in range(Nt+1):
        mu_init[it] = mu_0
        u_init[it] = u_T

    mu, u = solve_HJBKFP_Picard(mu_init, u_init, tol)
    print 'mu = ', mu
    print '\n\n'
    print 'u = ', u

    fig, ax = plt.subplots()
    #ax = fig.add_subplot(111, projection = '3d')
    ax.margins(0.05)
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, mu[:, 0], label="mu[0]", color='black')
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, mu[:, 1], label="mu[1]", color='red', linestyle='--')
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, mu[:, 2], label="mu[2]", color='green', linestyle=':', linewidth=5)
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, mu[:, 3], label="mu[3]", color='blue', linestyle='-.')
    ax.legend()
    #ax.set_yscale('log')
    figname = basename + '_mu_evol.eps'
    plt.savefig(figname)
    # plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    #ax = fig.add_subplot(111, projection = '3d')
    ax.margins(0.05)
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, u[:, 0], label="u[0]", color='black')
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, u[:, 1], label="u[1]", color='red', linestyle='--')
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, u[:, 2], label="u[2]", color='green', linestyle=':', linewidth=5)
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, u[:, 3], label="u[3]", color='blue', linestyle='-.')
    t_c = total_cost_vec(mu)
    plt.plot(np.linspace(0,Nt,Nt+1)*Dt, t_c, label="total cost", color='cyan')

    ax.legend()
    #ax.set_yscale('log')
    figname = basename + '_u_evol.eps'
    plt.savefig(figname)
    # plt.show()
    plt.close(fig)

    filename = basename + '_data_and_solution.npz'
    np.savez(filename, T=T, Nt=Nt, NS=NS, beta_UU=beta_UU, beta_UD=beta_UD, beta_DU=beta_DU, beta_DD=beta_DD, \
        v_H=v_H, lambda_speed=lambda_speed, q_rec_D=q_rec_D, q_rec_U=q_rec_U, q_inf_D=q_inf_D, q_inf_U=q_inf_U, \
        k_D=k_D, k_I=k_I, \
        mu=mu, u=u, t_c=t_c)
