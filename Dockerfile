# This is the Dockerfile that defines my development environment 
# for predictables. It is based on the official Python 3.11 image
# and installs the required packages from the requirements.txt file.

# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container to root
WORKDIR /

# Add the current directory contents into the container at root
ADD . /

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

# Create a virtual environment and activate it
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install any needed packages specified in requirements.txt to the virtual environment
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run bash when the container launches
CMD ["/bin/bash"]