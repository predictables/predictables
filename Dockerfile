# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container to /app
WORKDIR /home/app

# Add the current directory contents into the container at /home/app
ADD . /home/app

# Update the package list and install software
RUN apt-get update && apt-get upgrade -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    python3-cffi 
        
# Install any needed packages specified in requirements.txt
# RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Make port 80 available to the world outside this container
EXPOSE 80

# Run bash when the container launches
CMD ["/bin/bash"]