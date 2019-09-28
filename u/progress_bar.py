from __future__ import print_function, absolute_import
import enlighten

progress_manager = enlighten.get_manager()
active_counters = []

class Progress(object):

    def __init__(self, total, desc='', leave=False):
        self.counter = progress_manager.counter(total=total, desc=desc, leave=leave)
        active_counters.append(self.counter)

    def __iter__(self):
        return self
    
    def __next__(self):
        raise NotImplementedError()
    
    def close(self):
        self.counter.close()
        if self.counter in active_counters:
            active_counters.remove(self.counter)
        if len(active_counters) == 0:
            progress_manager.stop()

    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

class RangeProgress(Progress):
    def __init__(self, start, end, step=1, desc='', leave=False):
        self.i = start
        self.start = start
        self.end = end
        self.step = step
        super(RangeProgress, self).__init__((end - start) // step, desc=desc, leave=leave)
    
    def __next__(self):
        if self.i != self.start:
            self.counter.update()
        if self.i == self.end:
            self.close()
            raise StopIteration()
        i = self.i
        self.i += self.step
        return i

class ListProgress(RangeProgress):
    def __init__(self, li, desc='', leave=False):
        self.list = li
        super(ListProgress, self).__init__(0, len(self.list), desc=desc, leave=leave)
    
    def __next__(self):
        index = super(ListProgress, self).__next__()
        return self.list[index]

