#FROM python:3.9.19-bookworm
FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3

# Set working directory
WORKDIR /Main_gancxr

# Install additional packages
RUN apt-get update && \
    apt-get install -y git git-lfs libgl1 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib opencv-python scikit-image seaborn

# Run your script
CMD ["bash"]