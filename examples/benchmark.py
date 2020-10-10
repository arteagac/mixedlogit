import numpy as np
import pandas as pd
from timeit import default_timer as timer
import resource

import os
import psutil

def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (usage/1024)/1024
print("Initial: "+str(memory_usage()))


# ======= MixedLogit Library
import sys
sys.path.append(".")  # Path of mixedlogit library root folder.
from mixedlogit import MixedLogit

# Read and setup Electricity data
df = pd.read_csv("examples/data/electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
X = df[varnames].values
y = df['choice'].values

np.random.seed(0)
print("=== Electricity dataset. mixedlogit(GPU) ===")
print("{:7} {:4}  {:7} {:10} {} ".format("Ndraws", "Iter", "Time(s)", "LogLik",
                                         "RAM(mb)", "GPU(mb)", "Conver."))
for i in range(0, 2):
    start = timer()
    n_draws = (i+1)*100
    model = MixedLogit()
    model.fit(X, y, varnames, alternatives=[1, 2, 3, 4], n_draws=n_draws,
              asvars=varnames, mixby=df.id.values, verbose=0,
              randvars={'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n',
                        'tod': 'n', 'seas': 'n'})
    ellapsed = timer() - start
    print("{:7} {:4} {:7.2f} {:7.2f} {} ".format(
        n_draws, model.total_iter, ellapsed, model.loglikelihood,
        model.convergence))
    print("Initial: "+str(memory_usage()))
