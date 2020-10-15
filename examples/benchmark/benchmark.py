"""
This file executes the benchmark. Check the README.md file
to make sure all the requirments are satisfied.
"""

import os
from tools import init_profiler_output_file
init_profiler_output_file()


def profile_range_draws(command, r_draws, dataset, usegpu=False):
    print("\n\n=== "+dataset+" dataset. "+command.split()[1] +
          ('(using GPU)' if usegpu else '')+" ===")
    print("Ndraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")
    for r in range(1, r_draws+1):
        os.system("{} {} {} {} prof".format(command, r*100, dataset, usegpu*1))


def print_estimates(command, n_draws, dataset):
    print("\n\n=== "+dataset+" dataset. "+command.split()[1]+" ===")
    os.system("{} {} {} {} estim".format(command, n_draws, dataset, 0))


r_draws = 15
# Run profiling

profile_range_draws("python mixedlogit_run.py", r_draws, "artificial", True)
profile_range_draws("python mixedlogit_run.py", r_draws, "artificial")
profile_range_draws("python pylogit_run.py", r_draws, "artificial")
profile_range_draws("Rscript mlogit_run.R", r_draws, "artificial")
profile_range_draws("python mixedlogit_run.py", r_draws, "electricity", True)
profile_range_draws("python mixedlogit_run.py", r_draws, "electricity")
profile_range_draws("python pylogit_run.py", r_draws, "electricity")
profile_range_draws("Rscript mlogit_run.R", r_draws, "electricity")

# Print estimates

print_estimates("python mixedlogit_run.py", 200, "artificial")
print_estimates("python pylogit_run.py", 200, "artificial")
print_estimates("Rscript mlogit_run.R", 200, "artificial")
print_estimates("python mixedlogit_run.py", 700, "electricity")
print_estimates("python pylogit_run.py", 700, "electricity")
print_estimates("Rscript mlogit_run.R", 700, "electricity")

# Plot profiling results
os.system("python plot_profiling_results.py")
