import numpy as np
import pandas as pd
from tools import Profiler, curr_ram
import sys
sys.path.append("../../")  # Path of mixedlogit library root folder.
from mixedlogit import MixedLogit
from mixedlogit import device

data_folder = "https://raw.githubusercontent.com/arteagac/mixedlogit/master/"\
              "examples/data/"

df = pd.read_csv(data_folder+"artificial_long.csv")

varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr', 'emipp',
            'nonsig1', 'nonsig2', 'nonsig3']
X = df[varnames].values
y = df['choice'].values

ini_ram = curr_ram()

np.random.seed(0)
device.disable_gpu_acceleration()
print("\n\n=== Artificial dataset. mixedlogit ===")
print("Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
for i in range(0, 15):
    profiler = Profiler().start()
    n_draws = (i+1)*100
    model = MixedLogit()
    model.fit(X, y, varnames, alternatives=[1, 2, 3], n_draws=n_draws,
              asvars=varnames, verbose=0,
              randvars={'meals': 'n', 'petfr': 'n', 'emipp': 'n'})
    ellapsed, max_ram, max_gpu = profiler.stop()
    print("{:6} {:7.2f} {:11.2f} {:7.3f} {:7.3f} {}"
          .format(n_draws, ellapsed, model.loglikelihood,
                  max_ram - ini_ram, 0, model.convergence))
