FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3 python3-pip curl git
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin/"