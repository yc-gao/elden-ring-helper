import ctypes
import time
from multiprocessing import Process, Manager


class CommonAction:
    def __init__(self, period):
        self.period = period
        self.process = None
        self.flag = Manager().Value(ctypes.c_bool, False)

    def main_loop(self):
        cur_tm = time.time()
        while self.flag.value:
            next_tm = cur_tm + self.period
            self.run()
            dur_sleep = next_tm - time.time()
            if dur_sleep > 0:
                time.sleep(dur_sleep)
            else:
                next_tm = time.time()
            cur_tm = next_tm

    def run(self):
        pass

    def start(self):
        assert self.process is None, "Action already start"
        self.process = Process(target=self.main_loop)
        self.flag.value = True
        self.process.start()

    def stop(self):
        assert self.process is not None, "Invalid action"
        self.flag.value = False

    def join(self):
        self.process.join()
        self.process = None
        self.flag.value = False
