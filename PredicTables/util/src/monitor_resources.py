import psutil


def monitor_resources():
    """
    Monitor CPU and memory usage. Prints a message to the console
    informing the user of the current CPU and memory usage.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
