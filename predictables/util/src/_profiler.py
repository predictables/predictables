import cProfile
import datetime
import io
import os
import pstats
from typing import Callable

from dotenv import load_dotenv

from predictables.util.src.logging._DebugLogger import DebugLogger

load_dotenv()


def profiler(func: Callable) -> Callable:
    """Decorator to CPU and memory profile a Python function."""
    # Only profile if the LOGGING_LEVEL environment variable is set to debug
    if os.getenv("LOGGING_LEVEL", "info").lower() == "debug":
        function_name = func.__name__
        current_file = os.path.basename(__file__)
        dbg = DebugLogger(filename="performance.log", working_file=current_file)

        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            # Execute the function
            result = func(*args, **kwargs)

            profiler.disable()

            # Profiling for CPU
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
            stats.print_stats()

            # Prepare the profiling report
            profile_string = (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
            profile_string += f"Function: {function_name} | "
            profile_string += f"Working file: {current_file} | "
            profile_string += f"Arguments: {args} | "
            profile_string += f"Keyword arguments: {kwargs} | "
            profile_string += f"Result: {result} | "
            profile_string += "Profiler stats:\n"
            profile_string += stream.getvalue()
            dbg.msg(profile_string)
            return result

    else:

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

    return wrapper
