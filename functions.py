import numpy as np
from scipy.stats import uniform
    
def basic_metropolis_hastings(X0, max_t_iterations=10**3):
    """very simple version for this example 
    
    assumes 1 dimension in X"""

    def log_unnormalised_target_pdf(x):
        """standard normal in this example"""
        return -(x**2)/2
    
    def log_proposal_pdf(x, conditional):
        """uniform on (conditional-1/2, conditional+1/2)
        , so its pdf is 1 for all x in that interval
        """
        return 0
    
    def proposal_sample(conditional):
        """drawsa single sample from uniform on (conditional-1/2, conditional+1/2)"""
        return uniform.rvs(loc = conditional -.5, scale = 1)
    
    def log_alpha(current, new):
        top = log_unnormalised_target_pdf(new) + log_proposal_pdf(current, new)
        bottom = log_unnormalised_target_pdf(current) + log_proposal_pdf(new, current)
        return min( 0, top - bottom )
    
    chain = np.zeros(max_t_iterations)
    X_t = chain[0] = X0
    
    log_unif_rvs = np.log(uniform.rvs(size = max_t_iterations))
    for t in range(1, max_t_iterations):
        #propose a move from Q
        proposed_value = proposal_sample(X_t)#sample Q(.|X_t)
        #sample a uniform and take log
        log_u = log_unif_rvs[t]
        #get alpha on log scale
        log_alpha_prob = log_alpha(X_t, proposed_value)
        #decide if the chain accepts or rejects the move
        #this is setting X_t+1 but no point in creating another variable
        if log_u <= log_alpha_prob:
            X_t = proposed_value 
        #record the new state of the chain
        chain[t] = X_t

    return chain