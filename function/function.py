import numpy as np
import pandas as pd

def test(a,b):
	return a+b

# Assignment 1
def compute_loglikelihood_a1(b, df):
    p_i1 = np.exp(0)/(np.exp(0)+np.exp(b))
    y_to_matrix = np.reshape(np.array(df['y']),(-1,2))
    log_lh = (sum(y_to_matrix[:,0]*np.log(p_i1))+sum(y_to_matrix[:,1]*np.log(1-p_i1)))/1000
    return log_lh

# Assignment 2
def log_production(l, k, omega, eta, beta_0, beta_l, beta_k):
    y = beta_0 + beta_l * l + beta_k * k + omega + eta
    return y

def log_labor_choice(k, wage, omega, beta_0, beta_l, beta_k, sigma_eta):
    l = ((beta_l * np.exp(beta_0 + omega + sigma_eta**2 / 2) * (np.exp(k) ** beta_k)) / wage) ** (1 / (1 - beta_l))
    return np.log(l)

def log_labor_choice_error(k, wage, omega, beta_0, beta_l, beta_k, iota, sigma_eta):
    l_error = ((beta_l * np.exp(beta_0 + omega + iota + sigma_eta**2 / 2)\
                * (np.exp(k) ** beta_k)) / wage) ** (1 / (1 - beta_l))
    return np.log(l_error)

def investment_choice(k, omega, gamma, delta):
    invest = (delta + gamma * omega) * np.exp(k)
    return invest

def generate_new_df_a2(df, t, wage, gamma, alpha, delta, sigma_nu, sigma_iota, sigma_eta, \
                            beta_0, beta_l, beta_k):
    k = np.log((1 - delta) * np.exp(df['k']) + df['inv'])
    j = np.arange(1,1001)
    nu = np.random.normal(0, sigma_nu, 1000)
    omega = alpha * df['omega'] + nu
    df_new = pd.DataFrame({'j': j,'t': np.zeros(1000) + t, 'k': k, 'omega': omega, 'wage': np.zeros(1000) + wage})
    
    iota = np.random.normal(0, sigma_iota, 1000)
    df_new['iota'] = iota
    
    l = log_labor_choice(k = df_new['k'], wage = df_new['wage'], omega = df_new['omega'], 
                        beta_0 = beta_0, beta_l = beta_l, beta_k = beta_k, sigma_eta = sigma_eta)
    l_error = log_labor_choice_error(k = df_new['k'], wage = df_new['wage'], omega = df_new['omega'], 
                                    beta_0 = beta_0, beta_l = beta_l, beta_k = beta_k,
                                    iota = df_new['iota'], sigma_eta = sigma_eta)
    inv = investment_choice(k = df_new['k'], omega = df_new['omega'], gamma = gamma, delta = delta)
    df_new['l'] = l
    df_new['l_error'] = l_error
    df_new['inv'] = inv
    
    eta = np.random.normal(0, sigma_eta, 1000)   
    df_new['eta'] = eta
    
    y = log_production(l = df_new['l'], k = df_new['k'], omega = df_new['omega'], eta = df_new['eta'], \
                       beta_0 = beta_0, beta_l = beta_l, beta_k = beta_k)
    y_error = log_production(l = df_new['l_error'], k = df_new['k'], omega = df_new['omega'], \
                             eta = df_new['eta'], beta_0 = beta_0, beta_l = beta_l, beta_k = beta_k)
    df_new['y'] = y
    df_new['y_error'] = y_error
    df_new['nu'] = nu

    return df_new

def moment_olleypakes_2nd(alpha, beta_0, beta_k, df_all, df_all_1st):
    df_new = pd.DataFrame({'j':df_all['j'], 't':df_all['t'] + 1, 'k_t_1': df_all['k'], 'inv_t_1': df_all['inv']})
    df_new = pd.merge(df_all_1st, df_new, how='left', on=['j','t']).sort_values(by=['j','t'], ascending=True)
    df_new = pd.merge(df_all, df_new, how='left', on=['j','t']).sort_values(by=['j','t'], ascending=True)
    
    k_mean = ((df_new['y_error_tilde']-beta_0 - beta_k * df_new['k'] - \
              alpha * (df_new['phi_t_1'] - beta_0 - beta_k * df_new['k_t_1'])) * df_new['k']).mean()
    k_t_1_mean = ((df_new['y_error_tilde']-beta_0 - beta_k * df_new['k'] - \
              alpha * (df_new['phi_t_1'] - beta_0 - beta_k * df_new['k_t_1'])) * df_new['k_t_1']).mean()    
    inv_t_1_mean = ((df_new['y_error_tilde']-beta_0 - beta_k * df_new['k'] - \
              alpha * (df_new['phi_t_1'] - beta_0 - beta_k * df_new['k_t_1'])) * df_new['inv_t_1']).mean()  
    return np.array([k_mean, k_t_1_mean, inv_t_1_mean])

def objective_olleypakes_2nd(theta, df_all, df_all_1st, W):
    g = moment_olleypakes_2nd(alpha = theta[0], beta_0 = theta[1], beta_k = theta[2],\
                              df_all = df_all, df_all_1st = df_all_1st)
    q = np.dot(np.dot(g,W), g[np.newaxis].T)
    return(q[0]) 