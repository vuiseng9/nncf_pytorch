FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        git && \
rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        vim \
        wget \
        curl \
        zip \
        unzip \
        nano \
        openssh-server \
        openssh-client \
        sudo \
        graphviz libgraphviz-dev libgl1-mesa-glx \
    && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/miniconda/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda --version && python --version

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN git clone https://github.com/vuiseng9/nncf && \
    cd nncf && git checkout paas && \
    pip install -e . \
    pip install -r examples/requirements.txt && \
    pip install ipython jupyterlab pygraphviz more_itertools && \
    rm -rf ~/.cache/pip/* && \
    ninja_executable=$(python -c "import ninja; print(ninja.__path__[0])")/data/bin/ninja && \
    chmod +x $ninja_executable && \
    python -c "import nncf; print(\"NNCF {} is installed\".format(nncf.__version__))"