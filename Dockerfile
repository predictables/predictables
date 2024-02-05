# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container to /app
WORKDIR /

# Add the current directory contents into the container at /predictables
ADD . /predictables

# Update the package list and install software
RUN apt-get update && apt-get upgrade -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    python3-cffi \
    git \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg62-turbo-dev \
    vim \
    && apt-get clean
    

# Set the working directory in the container to /predictables
WORKDIR /predictables
        
# Install any needed packages specified in requirements.txt
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Make port 80 available to the world outside this container
EXPOSE 80

# Run bash when the container launches
CMD ["/bin/bash"]