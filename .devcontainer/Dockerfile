FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -y install \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-lang-portuguese \
    latexmk \
    biber \
    build-essential \
    python3 \
    python3-venv \
    python3-pip \
    && apt clean && rm -rf /var/lib/apt/lists/*