# The environment of the paper's trainings (henningrose/omnix:1.0.0), dependencies only.
#   docker build -t omnix docker/
FROM docker.io/pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# the base image's Python is the system Python
ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
