FROM ubuntu:bionic
LABEL maintainer="UVA Research Computing <uvarc@virginia.edu>"

RUN apt-get update
RUN apt-get -y upgrade
RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata
RUN apt-get install -y python3 python3-dev python3-pip nginx python3-opencv
RUN pip3 install uwsgi

ADD ./ ./app
WORKDIR ./app

RUN pip3 install -r requirements.txt

COPY ./nginx.conf /etc/nginx/sites-available/default
COPY ./fileserve.conf /etc/nginx/sites-available/fileserve

CMD ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default
CMD ln -s /etc/nginx/sites-available/fileserve /etc/nginx/sites-enabled/fileserve

CMD service nginx start && uwsgi -s /tmp/uwsgi.sock --chmod-socket=666 --manage-script-name --mount /=app:app
