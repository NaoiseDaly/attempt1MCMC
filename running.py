from functions import *
import logging
#set up logging to view chain execution times 
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)


first_chain = basic_metropolis_hastings(-5, 6000)

simply_plot_the_chain(first_chain, [100, 200, 300] )