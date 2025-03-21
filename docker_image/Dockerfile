FROM ubuntu

RUN apt-get update && apt-get install -y python3 python3-pip curl git ffmpeg texlive-latex-extra cm-super dvipng ghostscript
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin/"