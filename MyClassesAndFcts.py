import numpy as np

class AcquisitionFunction(object):
    
    def __init__(
        self,
        params,
        dt,
        t,
        q,
        seed
    ):

        self.params = params
        self.dt = dt
        self.t = t
        self.q = q
        
class RetentionFunction(object):
    
    def __init__(
        self: np.ndarray,
        params: np.ndarray,
        dt: float,
        t: np.ndarray,
        q: np.ndarray,
        seed: int
    ):

        self.params = params
        self.dt = dt
        self.t = t
        self.q = q
        
        
def WeibullGammaAcq(params, dt, t, q = 0, cov_mkt = None, seed=None, log_alpha = False)->AcquisitionFunction:
    """Simulates the Weibull model for a specified time duration
        ----------
        params : np.array, 1d of length dim_param
            Parameter vector
        dt : float
            Timestep
        t : array
            Numpy array with the time steps
        seed : int
        """
    #print(params)
    par_pop = params[0,0]
    par_pop.astype(float)
    par_lambda = params[0,1]  
    par_lambda.astype(float)
    par_c = params[0,2]  
    par_c.astype(float)
    par_r = params[0,3]  
    par_r.astype(float)
    par_beta_q = params[0,4]  
    par_beta_q.astype(float)
    par_beta_mkt = params[0,5]  
    par_beta_mkt.astype(float)
    
    if log_alpha:
        par_lambda = np.exp(par_lambda)
    #print(par_pop)
    
    q = np.array(q).astype(int)
    
    if cov_mkt is None:
        cov_mkt = np.ones_like(t)
        cov_mkt.astype(float)
        
    #cov_mkt_hf = np.array([0.018171081,0.036342162,0.054513243,0.072684324,0.090855405,0.109026486,0.127197568,0.145368649,0.16353973,0.181710811,0.199881892,0.218052973,0.236224054,0.254395135,0.272566216,0.290737297,0.308908378,0.327079459,0.345250541,0.363421622,0.381592703,0.399763784,0.417934865,0.436105946,0.627393493,0.81868104,1.009968586,1.201256133,1.39254368,1.583831227,1.775118773,1.96640632,2.157693867,2.348981414,2.54026896,2.731556507,3.645778254,4.56,5.474221746,6.112110873,6.75,7.387889127,8.808944563,10.23,11.65105544,13.13112772,14.6112,16.09127228,14.28563614,12.48,10.67436386,11.17218193,11.67,12.16781807,12.06890904,11.97,11.87109096,11.48764548,11.1042,10.72075452,15.05037726,19.38,23.70962274,20.62981137,17.55,14.47018863])    
    # qrt to assign quartly covariate
    
    idx = np.arange(t.size)

    mnth = np.mod(idx,12)+1
    qrt = np.floor_divide((mnth-1), 3)+1
        
    qrt_bool = (qrt<=np.max(q)) * (qrt>=np.min(q))
    idd_q = np.argwhere(qrt_bool)
    idd_nq = np.argwhere(np.logical_not(qrt_bool))
    
    cov_q = np.zeros_like(cov_mkt)

    cov_q[idd_q] = np.exp(par_beta_mkt*np.log(cov_mkt[idd_q])+par_beta_q)
    
    cov_q[idd_nq] = np.exp(par_beta_mkt*np.log(cov_mkt[idd_nq]))
    
    noise = 0.0
    
    tstep = float(dt)

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    ####################################
    # kinetics
    
    x = t

    ####################################
    # simulation from initial point
    
    #print(cov_q)
    b = np.copy(cov_q)
    idx = idx + 1
    
    ### Check indexing (cov is +1)
    for i in range(1,b.size):
        b[i] = b[i-1] + (idx[i]**par_c-idx[i-1]**par_c)*cov_q[i]
        
    
    V = (1.0-par_pop)*(1-(par_lambda/(par_lambda+b))**par_r)
    V0 = V[0]
    V[1:]=V[1:]-V[0:(V.size-1)]
    V[0] = V0

    return np.array(V).reshape(-1, 1)


def WeibullAcq(params, dt, t, q = 0, cov_mkt = None, seed=None, log_alpha = False)->AcquisitionFunction:
    """Simulates the Weibull model for a specified time duration
        ----------
        params : np.array, 1d of length dim_param
            Parameter vector
        dt : float
            Timestep
        t : array
            Numpy array with the time steps
        seed : int
        """
    #print(params)
    par_pop = params[0,0]
    par_pop.astype(float)
    par_lambda = params[0,1]  
    par_lambda.astype(float)
    par_c = params[0,2]  
    par_c.astype(float)
    #par_r = params[0,3]  
    #par_r.astype(float)
    par_beta_q = params[0,3]  
    par_beta_q.astype(float)
    par_beta_mkt = params[0,4]  
    par_beta_mkt.astype(float)
    
    if log_alpha:
        par_lambda = np.exp(par_lambda)
    #print(par_pop)
    
    q = np.array(q).astype(int)
    
    if cov_mkt is None:
        cov_mkt = np.ones_like(t)
        cov_mkt.astype(float)
        
    #cov_mkt_hf = np.array([0.018171081,0.036342162,0.054513243,0.072684324,0.090855405,0.109026486,0.127197568,0.145368649,0.16353973,0.181710811,0.199881892,0.218052973,0.236224054,0.254395135,0.272566216,0.290737297,0.308908378,0.327079459,0.345250541,0.363421622,0.381592703,0.399763784,0.417934865,0.436105946,0.627393493,0.81868104,1.009968586,1.201256133,1.39254368,1.583831227,1.775118773,1.96640632,2.157693867,2.348981414,2.54026896,2.731556507,3.645778254,4.56,5.474221746,6.112110873,6.75,7.387889127,8.808944563,10.23,11.65105544,13.13112772,14.6112,16.09127228,14.28563614,12.48,10.67436386,11.17218193,11.67,12.16781807,12.06890904,11.97,11.87109096,11.48764548,11.1042,10.72075452,15.05037726,19.38,23.70962274,20.62981137,17.55,14.47018863])    
    # qrt to assign quartly covariate
    
    idx = np.arange(t.size)

    mnth = np.mod(idx,12)+1
    qrt = np.floor_divide((mnth-1), 3)+1
        
    qrt_bool = (qrt<=np.max(q)) * (qrt>=np.min(q))
    idd_q = np.argwhere(qrt_bool)
    idd_nq = np.argwhere(np.logical_not(qrt_bool))
    
    cov_q = np.zeros_like(cov_mkt)

    cov_q[idd_q] = np.exp(par_beta_mkt*np.log(cov_mkt[idd_q])+par_beta_q)
    
    cov_q[idd_nq] = np.exp(par_beta_mkt*np.log(cov_mkt[idd_nq]))
    
    noise = 0.0
    
    tstep = float(dt)

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    ####################################
    # kinetics
    
    x = t

    ####################################
    # simulation from initial point
    
    #print(cov_q)
    b = np.copy(cov_q)
    idx = idx + 1
    
    ### Check indexing (cov is +1)
    for i in range(1,b.size):
        b[i] = b[i-1] + (idx[i]**par_c-idx[i-1]**par_c)*cov_q[i]
        
    
    V = (1.0-par_pop)*(1-np.exp(- par_lambda * b))
    V0 = V[0]
    V[1:]=V[1:]-V[0:(V.size-1)]
    V[0] = V0

    return np.array(V).reshape(-1, 1)

def WeibullGammaRet(params, dt, t, q = 0, seed = None, log_alpha = False)->RetentionFunction:
    """Simulates the Weibull model for a specified time duration
        ----------
        params : np.array, 1d of length dim_param
            Parameter vector
        dt : float
            Timestep
        t : array
            Numpy array with the time steps
        seed : int
        """
    par_alpha = params[0, 0]  
    par_alpha.astype(float)
    par_c = params[0, 1]  
    par_c.astype(float)
    par_r = params[0, 2]  
    par_r.astype(float)
    par_beta_q = params[0, 3]  
    par_beta_q.astype(float)
    
    q = np.array(q).astype(int)
    
    if log_alpha:
        par_alpha = np.exp(par_alpha)
        par_r = np.exp(par_r)
        
    # qrt to assign quartly covariate
    
    idx = np.arange(t.size)
    qrt = np.mod(idx,4)+1
    
    # quarter with active quarterly covariate
    if q.size > 4:
        cov_q = np.exp(q*par_beta_q)
    else:
        mnth = np.mod(idx,12)+1
        qrt = np.floor_divide((mnth-1), 3)+1

        qrt_bool = (qrt<=np.max(q)) * (qrt>=np.min(q))
        idd_q = np.argwhere(qrt_bool)
        idd_nq = np.argwhere(np.logical_not(qrt_bool))

        cov_q = np.zeros_like(t)
        cov_q[idd_q] = np.exp(par_beta_q)

        cov_q[idd_nq] = np.exp(0.0)

    
    noise = 0.0
    
    tstep = float(dt)

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    ####################################
    # kinetics


    ####################################
    # simulation from initial point
    
    idx += 1
    
    b = np.copy(cov_q)
    
    for i in range(1,b.size):
        b[i] = b[i-1] + (idx[i]**par_c-idx[i-1]**par_c)*cov_q[i]
    
    V = (par_alpha/(par_alpha+b))**par_r

    return np.array(V).reshape(-1, 1)

class CustomerBase:
    
    def new_customers(params, dt, t, q = 0, pop = 1, cov_mkt = None, seed = None,log_alpha = False, GammaMix = True):
        if GammaMix:
            ret = pop*WeibullGammaAcq(params, dt, t, q, cov_mkt, seed, log_alpha)
        else:
            ret = pop*WeibullAcq(params, dt, t, q, cov_mkt, seed, log_alpha)
        return ret
    
    def active_cohorts(params, dt, t, q = 0, new_customers = None, seed = None, log_alpha = False):
        
        cohort = np.zeros((t.size,t.size))
        
        if new_customers is None:
            new_customers = np.ones(t.size)
        
        np.fill_diagonal(cohort,new_customers)
        
        T = np.linspace(1,12,12)
        
        idx = np.arange(t.size)
        qrt = np.mod(idx,4)+1
        mnth = np.mod(idx,12)+1
        qrt_mon = mnth[qrt == q]
        
        idx = np.arange(t.size)

        mnth = np.mod(idx,12)+1
        qrt = np.floor_divide((mnth-1), 3)+1
        
        qrt_bool = (qrt<=np.max(q)) * (qrt>=np.min(q))
        idd_q = np.argwhere(qrt_bool)
        idd_nq = np.argwhere(np.logical_not(qrt_bool))

        cov_q = np.zeros_like(t)
        cov_q[idd_q] = 1.0
        #print(cov_q)
        #cov_q[idd_nq] = 0.0
        qm = np.copy(cov_q)
        qm = np.concatenate((qm,qm[mnth[t.size-1]-1:12]),axis=None)
        
            
        for i in range(0,t.size):
            m = mnth[i].astype(int)
            ## Usually cov_q[(m-1):(t.size-m)] but retention goes ohne month early
            qm = np.roll(qm,-1) ## Problem with roll => incorrect length /12
            #print(qm,qm.size)
            retention =  WeibullGammaRet(params, dt, t, qm[0:t.size], seed, log_alpha)
            
            cohort[i,(i+1):t.size] = new_customers[i]*retention[0:(t.size-i-1)].reshape(1,-1)
        
        return cohort
          
    def active_users_qrtly(params_acq,params_ret, dt, t, pop, q_acq,q_ret,cov_mkt, active = 3, seed = None, log_alpha = False, GammaMix = True):
        new_customers = CustomerBase.new_customers(params_acq, dt, t, q = q_acq, pop = pop, cov_mkt = cov_mkt, seed = seed,log_alpha = log_alpha, GammaMix = GammaMix)
        active_mat = CustomerBase.active_cohorts(params_ret, dt, t, q = q_ret, new_customers = new_customers, seed  = seed, log_alpha = log_alpha)
        qrt = 3
        act = active
        active_cust = np.zeros(np.ceil(t.size/qrt).astype(int))
        new_cust = np.copy(active_cust)

        for i in range(0,t.size):
            k = np.floor(i/qrt).astype(int)
            j = np.floor(i/12).astype(int)
            new_cust[k] += active_mat[i,i]
            if(np.mod(i+1,qrt) == 0):
                #print(i,k)
                a =  np.sum(active_mat[0:max(i-act+1,0),max(i-act+1,0)])
                dia = (np.arange(min(act,i+1))+max(i+1-act,0),np.arange(min(act,i+1))+max(i+1-act,0))
                b =  np.sum(active_mat[dia])
                active_cust[k] = a + b
                #print(dia,np.sum(active_mat[dia]))
                #print(max(i-act+1,0),a)
        return active_cust