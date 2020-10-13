import numpy as np
import pandas as pd
from collections import OrderedDict
import pylogit as pl
import io
import sys
from tools import Profiler, curr_ram


data_folder = "https://raw.githubusercontent.com/arteagac/mixedlogit/master/"\
              "examples/data/"
df = pd.read_csv(data_folder+"electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
spec, spec_names = OrderedDict(), OrderedDict()

for col in varnames:
    df[col] = df[col].astype(float)
    spec[col] = [[1, 2, 3, 4]]
    spec_names[col] = [col]

ini_ram = curr_ram()

np.random.seed(0)
print("\n\n=== Electricity dataset. pylogit ===")
print("Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
for i in range(0, 5):
    profiler = Profiler().start()
    n_draws = (i+1)*100
    # Prints are temporarily disabled as pylogit has excessive verbosity
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()  # Disable print
    model = pl.create_choice_model(data=df, alt_id_col="alt",
                                   obs_id_col="chid", choice_col="choice",
                                   specification=spec,
                                   model_type="Mixed Logit", names=spec_names,
                                   mixing_id_col="id", mixing_vars=varnames)
    model.fit_mle(init_vals=np.zeros(2 * len(varnames)),
                  num_draws=n_draws, seed=123)
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__  # Enable print
    ellapsed, max_ram, max_gpu = profiler.stop()
    print("{:6} {:7.2f} {:11.2f} {:7.3f} {:7.3f} {}"
          .format(n_draws, ellapsed, model.log_likelihood,
                  max_ram - ini_ram, 0, model.estimation_success))
