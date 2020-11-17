#!/bin/bash
# Usage: Use commands `./app.sh build`, `./app.sh run`, `./app.sh exec`, and `./app.sh stop`.

if [ $1 == "build" ]
then
    docker build -t stormlight .
    # Need to download data when running for the first time on another machine.
    if [ ! -d data ]
    then
        curl -L -o tmp.zip https://www.dropbox.com/sh/y4i0hd2bs7cgfk8/AABjyZj77X-qRWKcoYHBxyvla?dl=1
        unzip tmp.zip -d data
        rm tmp.zip
    fi
elif [ $1 == "run" ]
then
    VOLUME_PATH=$(cd data && pwd)
    docker run --rm --name stormlight-app -v $VOLUME_PATH:/stormlight/data -p 8080:8080 stormlight
elif [ $1 == "stop" ]
then
    docker stop stormlight-app
elif [ $1 == "exec" ]
then
    docker exec -it stormlight-app /bin/bash
elif [ $1 == "colab" ]
then
    pip install -r requirements.txt
    pip install -e lib
    sudo apt update && sudo apt install curl
    curl -L -o tmp.zip https://www.dropbox.com/sh/y4i0hd2bs7cgfk8/AABjyZj77X-qRWKcoYHBxyvla?dl=1
    unzip tmp.zip -d data -x /
    rm tmp.zip
    pip install pyngrok
else
    echo 'Invalid command. Options are "build", "stop", "run", and "exec".'
fi

