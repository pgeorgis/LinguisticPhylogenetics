FROM python:3.12-alpine

RUN apk update && apk upgrade && apk add --no-cache R git make bash
