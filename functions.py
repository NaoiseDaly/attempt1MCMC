import numpy as np
from scipy.stats import uniform
from scipy.special import loggamma
import matplotlib.pyplot as plt
import logging 
from time import perf_counter
logger = logging.getLogger(__name__) #could output to file here

    
def basic_metropolis_hastings(X0, max_t_iterations=10**3):
    """very simple version for this example 
    
    assumes 1 dimension in X"""

    #start timing here
    start_time = perf_counter()

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

    #end timing now
    end_time = perf_counter()
    #record timing
    logger.info(
        f"chain took {round(end_time-start_time,3)} secs to simulate {max_t_iterations} iterations"
    )

    return chain

def simply_plot_the_chain(chain, with_burn_in = None, fmt_plt = "-"):
    """plot the chain over time
    
    optionally view the chain after different burn in points,
    it would be preferably to pick an odd number of burn in points"""
    if not with_burn_in:
        fig, ax = plt.subplots()
        ax.plot(chain, fmt_plt)
        ax.set_xlabel("t")
        ax.set_ylabel("X")
        plt.show()
        return
    
    with_burn_in.insert(0, 0) #show the whole chain for reference

    num_subplots = len(with_burn_in)
    num_rows  = (num_subplots+1) //2
    fig, axes = plt.subplots( nrows = num_rows, ncols = 2)
    axes = np.array(axes).flatten() # easier to access subplots this way

    
    for burn_in_point, subplot in zip(with_burn_in, axes):
        subplot.plot(chain[burn_in_point:], fmt_plt)
        subplot.set_xlabel("t")
        subplot.set_ylabel("X")
        subplot.set_title(f"burn in after {burn_in_point}")
    plt.show()


def poisson_3_MCMC(X0, max_t_iterations=10**3):
    """very simple version for this example 
    
    assumes 1 dimension in X"""

    #start timing here
    start_time = perf_counter()

    def log_unnormalised_target_pdf(x):
        """poisson with rate 3 in this example
        assumes x is an integer"""
        if x < 0:
            return -np.inf #log of zero
        elif x == 0:
            return -3 #getting overflows if i do ln(factorial(x)) but gamma(0) is infinity
        else:
            return -3 + (x*np.log(3)) - loggamma(x) # log of factorial of x
    
    def log_proposal_pdf(x, conditional):
        """steps +/- 1 from conditional with equal chance
        """
        if abs(x - conditional) == 1:
            return np.log(.5)
        else:
            return -np.inf #log of zero
    
    def proposal_sample(conditional):
        """ steps +/- 1 from conditional with equal chance """
        if uniform.rvs() >= .5:
            return conditional+1
        else:
            return conditional-1
    
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

    #end timing now
    end_time = perf_counter()
    #record timing
    logger.info(
        f"chain took {round(end_time-start_time,3)} secs to simulate {max_t_iterations} iterations"
    )

    return chain