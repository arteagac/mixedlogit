import numpy as np
import pandas as pd
from timeit import default_timer as timer
import resource
from time import sleep
from threading import Thread
import cupy as cp
import sys
sys.path.append(".")  # Path of mixedlogit library root folder.
from mixedlogit import MixedLogit

cupymem = cp.get_default_memory_pool()
data_folder = "https://raw.githubusercontent.com/arteagac/mixedlogit/master/"\
              "examples/data/"

curr_ram = lambda: resource.getrusage(resource.RUSAGE_SELF
                                      ).ru_maxrss/(1024*1024)
curr_gpu = lambda: cupymem.total_bytes()/(1024*1024*1024)

ini_ram = curr_ram()
print("ini_ram={:.3f}".format(ini_ram))


class Profiler():
    max_ram = 0
    max_gpu = 0
    thread_running = True
    start_time = None

    def _measure(self, measure_gpu_mem):
        while self.thread_running:
            self.max_ram = max(self.max_ram, curr_ram())
            if measure_gpu_mem:
                self.max_gpu = max(self.max_gpu, curr_gpu())
            sleep(.1)

    def start(self, measure_gpu_mem=True):
        Thread(target=self._measure, args=(measure_gpu_mem,)).start()
        self.start_time = timer()
        return self

    def stop(self):
        self.thread_running = False  # Stop thread
        ellapsed = timer() - self.start_time
        return ellapsed, self.max_ram, self.max_gpu


# ======= MixedLogit Library
# Read and setup Electricity data
df = pd.read_csv(data_folder+"electricity_long.csv")

varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
X = df[varnames].values
y = df['choice'].values

np.random.seed(0)
print("=== Electricity dataset. mixedlogit(GPU) ===")
print("Ndraws Iter Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
for i in range(0, 3):
    profiler = Profiler().start()
    n_draws = (i+1)*100
    model = MixedLogit()
    model.fit(X, y, varnames, alternatives=[1, 2, 3, 4], n_draws=n_draws,
              asvars=varnames, mixby=df.id.values, verbose=0,
              randvars={'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n',
                        'tod': 'n', 'seas': 'n'})
    ellapsed, max_ram, max_gpu = profiler.stop()
    print("{:6} {:4} {:7.2f} {:11.2f} {:7.3f} {:7.3f} {:>}"
          .format(n_draws, model.total_iter, ellapsed, model.loglikelihood,
                  max_ram - ini_ram, max_gpu, model.convergence))


np.random.seed(0)
from mixedlogit import use_gpu_acceleration
use_gpu_acceleration(False)
print("=== Electricity dataset. mixedlogit(CPU) ===")
print("Ndraws Iter Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
for i in range(0, 3):
    profiler = Profiler().start(measure_gpu_mem=False)
    n_draws = (i+1)*100
    model = MixedLogit()
    model.fit(X, y, varnames, alternatives=[1, 2, 3, 4], n_draws=n_draws,
              asvars=varnames, mixby=df.id.values, verbose=0,
              randvars={'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n',
                        'tod': 'n', 'seas': 'n'})
    ellapsed, max_ram, max_gpu = profiler.stop()
    print("{:6} {:4} {:7.2f} {:11.2f} {:7.3f} {:7.3f} {:>}"
          .format(n_draws, model.total_iter, ellapsed, model.loglikelihood,
                  max_ram - ini_ram, max_gpu, model.convergence))