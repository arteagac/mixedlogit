import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/profiling_results.csv")

libs = ['pylogit', 'mlogit', 'mixedlogit', 'mixedlogit_gpu']


def plot_memory_benchmark(dataset):
    dfe = df[df.dataset == dataset]
    for lib in libs:
        d = dfe[dfe.library == lib][["draws", "ram"]].values.T
        plt.plot(d[0], d[1])
    d = dfe[dfe.library == "mixedlogit_gpu"][["draws", "gpu"]].values.T
    plt.plot(d[0], d[1])
    plt.legend([i + " (RAM)" for i in libs] + ["mixedlogit_gpu (GPU)"])
    plt.xlabel("Random draws")
    plt.ylabel("Memory usage (GB)")
    plt.title("Memory usage ("+dataset+" dataset)")
    plt.savefig("results/memory_benchmark_"+dataset, dpi=300)
    plt.show()


def plot_time_benchmark(dataset):
    dfe = df[df.dataset == dataset]
    for lib in libs:
        d = dfe[dfe.library == lib][["draws", "time"]].values.T
        plt.plot(d[0], d[1])
    plt.legend(libs)
    plt.xlabel("Random draws")
    plt.ylabel("Time (Seconds)")
    plt.title("Estimation time ("+dataset+" dataset)")
    plt.savefig("results/time_benchmark_"+dataset, dpi=300)
    plt.show()


plot_memory_benchmark("electricity")
plot_memory_benchmark("artificial")

plot_time_benchmark("electricity")
plot_time_benchmark("artificial")
