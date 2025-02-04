FROM python:3.12-alpine

RUN apk update && apk add --no-cache \
    R R-dev \
    git \
    bash \
	gcc g++ \
    make \
    libhdf5-dev
