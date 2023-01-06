# Build this image:  docker build -t mpi .

FROM ubuntu:20.04


MAINTAINER Mohamad Arafeh <arafeh198@gmail.com>

ENV USER mpirun

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/${USER} 


RUN apt-get update

RUN apt-get install -y --no-install-recommends sudo apt-utils

RUN apt-get install -y software-properties-common gcc && \
    add-apt-repository -y 'ppa:deadsnakes/ppa'

RUN apt-get update

RUN apt-get update && apt-get install -y python3.9 python3-distutils \
    python3-pip python3-apt git wget vim nano

RUN apt-get install -y --no-install-recommends openssh-server \
    gfortran libopenmpi-dev openmpi-bin openmpi-common openmpi-doc binutils
	
RUN mkdir /var/run/sshd
RUN echo 'root:${USER}' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# ------------------------------------------------------------
# Add an 'mpirun' user
# ------------------------------------------------------------

RUN adduser --disabled-password --gecos "" ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ------------------------------------------------------------
# Set-Up SSH with our Github deploy key
# ------------------------------------------------------------

ENV SSHDIR ${HOME}/.ssh/

RUN mkdir -p ${SSHDIR}

ADD utilities/mpi/ssh/config ${SSHDIR}/config
ADD utilities/mpi/ssh/id_rsa.mpi ${SSHDIR}/id_rsa
ADD utilities/mpi/ssh/id_rsa.mpi.pub ${SSHDIR}/id_rsa.pub
ADD utilities/mpi/ssh/id_rsa.mpi.pub ${SSHDIR}/authorized_keys

RUN chmod -R 600 ${SSHDIR}* && \
    chown -R ${USER}:${USER} ${SSHDIR}

# RUN pip3 install --upgrade python3-pip

USER ${USER}
RUN  pip3 install --user -U setuptools \
    && pip3 install --user mpi4py
    


# ------------------------------------------------------------
# Add localfed  lib
# ------------------------------------------------------------
ENV FedLib=${HOME}/localfed
ENV PYTHONPATH "${PYTHONPATH}:${FedLib}"

ADD . ${FedLib}

RUN pip3 install -r ${FedLib}/requirements.txt

# ------------------------------------------------------------
# Configure OpenMPI
# ------------------------------------------------------------

USER root

RUN rm -fr ${HOME}/.openmpi && mkdir -p ${HOME}/.openmpi
ADD  utilities/mpi/default-mca-params.conf ${HOME}/.openmpi/mca-params.conf
RUN chown -R ${USER}:${USER} ${HOME}/.openmpi
RUN chown -R ${USER}:${USER} ${FedLib}


# ------------------------------------------------------------
# Copy MPI4PY example scripts
# ------------------------------------------------------------

ENV TRIGGER 1

ADD utilities/mpi/benchmarks ${HOME}/mpi4py_benchmarks
RUN chown -R ${USER}:${USER} ${HOME}/mpi4py_benchmarks

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
