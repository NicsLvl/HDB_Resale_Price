from functools import wraps
import tracemalloc


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start} seconds \n")
        return result
    return wrapper


def memory_usage(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Function '{func.__name__}' Memory usage:")
        print(f"  Current: {current / 10**6:.4f} MB")
        print(f"  Peak: {peak / 10**6:.4f} MB \n")
        return result
    return wrapper
