FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

#  set up user with uid 1000 for vscode devcontainer
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # dev dependencies
    curl vim nano tar iputils-ping screen ffmpeg\
    # kenlm dependencies 
    build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev && \
    apt-get clean

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# set the default user
USER $USERNAME

# copy the rest of the files
COPY swr2_asr ./swr2_asr
COPY data ./data
COPY config.* ./

# just keep the container alive
ENTRYPOINT ["tail", "-f", "/dev/null"]
