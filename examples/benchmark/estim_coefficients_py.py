import numpy as np
import pandas as pd
from collections import OrderedDict
import pylogit as pl
import io
import sys
sys.path.append("../../")  # Path of mixedlogit library root folder.
from mixedlogit import MixedLogit

data_folder = "https://raw.githubusercontent.com/arteagac/mixedlogit/master/"\
              "examples/data/"

# ======= ELECTRICTY DATASET ========
df = pd.read_csv(data_folder+"electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]

def print_estimates(names, coeff, stderr):
    print("Variable    Estimate   Std.Err.")
    for i in range(len(names)):
        print("{:9}  {:9.5}  {:9.5}".format(names[i][:8], coeff[i], stderr[i]))


print("\n\n=== Electricity dataset. mixedlogit ===")
X = df[varnames].values
y = df['choice'].values
model = MixedLogit()
model.fit(X, y, varnames, alternatives=[1, 2, 3, 4], n_draws=600,
          asvars=varnames, mixby=df.id.values, verbose=0,
          randvars={'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n',
                    'tod': 'n', 'seas': 'n'})
print_estimates(model.coeff_names, model.coeff_, model.stderr)
print("Log.Lik:   {:9.2f}".format(model.loglikelihood))

print("\n\n=== Electricity dataset. pylogit ===")
spec, spec_names = OrderedDict(), OrderedDict()
for col in varnames:
    df[col] = df[col].astype(float)
    spec[col] = [[1, 2, 3, 4]]
    spec_names[col] = [col]
# Prints are temporarily disabled as pylogit has excessive verbosity
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()  # Disable print
model = pl.create_choice_model(data=df, alt_id_col="alt",
                               obs_id_col="chid", choice_col="choice",
                               specification=spec,
                               model_type="Mixed Logit", names=spec_names,
                               mixing_id_col="id", mixing_vars=varnames)
model.fit_mle(init_vals=np.zeros(2 * len(varnames)),
              num_draws=600, seed=123)
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__  # Enable print
summ = model.summary

print_estimates(summ.index.values, summ.parameters.values, summ.std_err.values)
print("Log.Lik:   {:9.2f}".format(model.log_likelihood))

# ======= ARTIFICIAL DATASET ========

df = pd.read_csv(data_folder+"artificial_long.csv")

varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr', 'emipp',
            'nonsig1', 'nonsig2', 'nonsig3']

print("\n\n=== Artificial dataset. mixedlogit ===")
X = df[varnames].values
y = df['choice'].values
model = MixedLogit()
model.fit(X, y, varnames, alternatives=[1, 2, 3], n_draws=200,
          asvars=varnames, verbose=0,
          randvars={'meals': 'n', 'petfr': 'n', 'emipp': 'n'})
print_estimates(model.coeff_names, model.coeff_, model.stderr)
print("Log.Lik:   {:9.2f}".format(model.loglikelihood))


print("\n\n=== Artificial dataset. pylogit ===")
spec, spec_names = OrderedDict(), OrderedDict()

for col in varnames:
    df[col] = df[col].astype(float)
    spec[col] = [[1, 2, 3]]
    spec_names[col] = [col]

# Prints are temporarily disabled as pylogit has excessive verbosity
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()  # Disable print
model = pl.create_choice_model(data=df, alt_id_col="alt",
                               obs_id_col="id", choice_col="choice",
                               specification=spec,
                               model_type="Mixed Logit", names=spec_names,
                               mixing_id_col="id", mixing_vars=varnames)
model.fit_mle(init_vals=np.zeros(2 * len(varnames)),
              num_draws=200, seed=123)
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__  # Enable print

print_estimates(summ.index.values, summ.parameters.values, summ.std_err.values)
print("Log.Lik:   {:9.2f}".format(model.log_likelihood))
