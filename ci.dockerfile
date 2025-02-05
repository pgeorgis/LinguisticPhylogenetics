FROM python:3.12-alpine

RUN apk update && apk add --no-cache \
    R R-dev R-doc \
    git \
    bash \
    gcc g++ make \
    hdf5-dev
