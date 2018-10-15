FROM dorowu/ubuntu-desktop-lxde-vnc:xenial
MAINTAINER denden047

# init
WORKDIR /opt
RUN apt-get update && apt-get upgrade -y

# python3 env
RUN apt-get install -y python3-pip
ENV python="python3"
ENV pip="python3 -m pip"
RUN python3 -m pip install -U pip

# tensorflow-cpu
RUN python3 -m pip install -U tensorflow
RUN python3 -c "import tensorflow as tf; print(tf.__version__)"

# pybullet
RUN python3 -m pip install -U pybullet

# gym
RUN apt-get install -y \
    python-pyglet \
    python3-opengl \
    zlib1g-dev \
    libjpeg-dev \
    patchelf \
    cmake \
    swig \
    libboost-all-dev \
    libsdl2-dev \
    libosmesa6-dev \
    xvfb \
    ffmpeg
RUN python3 -m pip install 'gym[all]'

# baselines
RUN python3 -m pip install baselines

# others
RUN python3 -m pip install matplotlib
RUN python3 -m pip install transforms3d
RUN python3 -m pip install pandas
RUN python3 -m pip install h5py==2.8.0rc1

# others
RUN python3 -m pip isntall keras
RUN python3 -m pip isntall pydot

# setting
ENTRYPOINT [""]
CMD ["/startup.sh"]
