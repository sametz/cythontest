from functools import wraps
import time


def timefn(fn):
    """Time the execution of the wrapped function.

    From "High Performance Python" by Micha Gorelick and Ian Ozsvald,
    O'Reilly Media, 2014
    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print((
            "@timefn:" + fn.__name__ + " took " + str(t2 - t1) + " seconds"))
        return result
    return measure_time
