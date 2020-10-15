import cupy
from time import sleep, time
from threading import Thread
import resource
import os
cupymem = cupy.get_default_memory_pool()
output_file = "results/profiling_results.csv"


def curr_ram():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)


def curr_gpu():
    return cupymem.total_bytes()/(1024*1024*1024)


def init_profiler_output_file():
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'a') as fw:
        fw.write("library,dataset,draws,time,loglik,ram,gpu,converg\n")


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
            sleep(.05)

    def start(self, measure_gpu_mem=False):
        Thread(target=self._measure, args=(measure_gpu_mem,)).start()
        self.start_time = time()
        return self

    def stop(self):
        self.thread_running = False  # Stop thread
        ellapsed = time() - self.start_time
        return ellapsed, self.max_ram, self.max_gpu

    def export(self, library, dataset,
               n_draws, ellapsed, loglik, ram, gpu, success):
        with open(output_file, 'a') as fw:
            fw.write("{},{},{},{},{},{},{},{}\n"
                     .format(library, dataset, n_draws, ellapsed, loglik,
                             ram, gpu, success))
