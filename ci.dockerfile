FROM python:3.12

ARG DEBIAN_FRONTEND=noninteractive

# Install R and other dependencies in Debian Bookworm
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    r-base \
    r-base-dev
