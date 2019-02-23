FROM centos:latest
ENV PACKAGES="\
    gcc \
    gcc-c++ \ 
    openssl-devel \ 
    bzip2-devel \
    zlib \
    zlib-devel \
    libffi-devel \ 
    wget tar \
    make \
    mysql-devel \ 
    gcc-devel \
    python3.6-devel \
    sudo \
    python3.6-setuptools \
    curl \
    wget \ 
    MySQL-python \ 
    unixODBC \
    unixODBC-devel \
    "
# PYTHON DATA SCIENCE PACKAGES
ENV PYTHON_PACKAGES="\
    wheel \ 
    setuptools \
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    nltk \
    tensorflow\
    sklearn\
    statsmodels\
    influxdb\
    keras\
    pyodbc\
    mysql\
    kafka-python\
    grafana-api\
    " 
RUN mkdir python_setup
WORKDIR ./python_setup
RUN yum -y update \
    && yum -y install $PACKAGES \
    && wget https://www.python.org/ftp/python/3.6.6/Python-3.6.6.tgz \
    && tar xzf Python-3.6.6.tgz \
    && rm Python-3.6.6.tgz \
    && cd Python-3.6.6/ \
    && ./configure --enable-optimizations \ 
    && make altinstall \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.6 get-pip.py \
    && python3.6 -m ensurepip \
    && python3.6 -m pip install --no-cache-dir $PYTHON_PACKAGES \
