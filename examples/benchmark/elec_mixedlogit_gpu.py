import numpy as np
import pandas as pd
from tools import Profiler, curr_ram
import sys
sys.path.append("../../")  # Path of mixedlogit library root folder.
from mixedlogit import MixedLogit

data_folder = "https://raw.githubusercontent.com/arteagac/mixedlogit/master/"\
              "examples/data/"

df = pd.read_csv(data_folder+"electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
X = df[varnames].values
y = df['choice'].values

ini_ram = curr_ram()

np.random.seed(0)
print("\n\n=== Electricity dataset. mixedlogit (GPU) ===")
print("Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
for i in range(0, 15):
    profiler = Profiler().start(measure_gpu_mem=True)
    n_draws = (i+1)*100
    model = MixedLogit()
    model.fit(X, y, varnames, alternatives=[1, 2, 3, 4], n_draws=n_draws,
              asvars=varnames, mixby=df.id.values, verbose=0,
              randvars={'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n',
                        'tod': 'n', 'seas': 'n'})
    ellapsed, max_ram, max_gpu = profiler.stop()
    print("{:6} {:7.2f} {:11.2f} {:7.3f} {:7.3f} {}"
          .format(n_draws, ellapsed, model.loglikelihood,
                  max_ram - ini_ram, max_gpu, model.convergence))
