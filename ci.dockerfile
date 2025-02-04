FROM python:3.12-alpine

# Update and install R and other dependencies in Alpine
RUN apk update && apk add --no-cache R R-dev
