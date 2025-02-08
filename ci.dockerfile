FROM python:3.12-alpine

RUN apk update && apk add --no-cache \
    R R-dev R-doc \
    git \
    bash \
    gcc g++ make \
    hdf5-dev \
    msttcorefonts-installer fontconfig && update-ms-fonts && fc-cache -f

WORKDIR /phyloLing
COPY install_r_dependencies.R install_r_dependencies.R
COPY phyloLing/utils/r/dependencies.R phyloLing/utils/r/dependencies.R
RUN ./install_r_dependencies.R
