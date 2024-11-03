from functions import *
import logging
#set up logging to view chain execution times 
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)


# first_chain = basic_metropolis_hastings(-5, 6000)

# simply_plot_the_chain(first_chain, [100, 200, 300] )

first_poisson_chain = poisson_3_MCMC(0, 6000)
simply_plot_the_chain(first_poisson_chain, fmt_plt="kD")
plt.bar(*np.unique(first_poisson_chain, return_counts=True))
plt.show()

second_poisson_chain = poisson_3_MCMC(50, 600)

simply_plot_the_chain(second_poisson_chain, fmt_plt="kD", with_burn_in= [40, 60, 100])
plt.bar(*np.unique(second_poisson_chain, return_counts=True))
plt.show()