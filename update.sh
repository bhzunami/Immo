#!/bin/bash

script() {
  cd /var/docker/container/Immo
  git pull
  cd immo
  docker-compose down
  docker-compose build
}

out=$(script 2>&1)
if [ $? -gt 0 ]; then
  echo Build failed!
  echo 
  echo $out # this syntax loses new lines somehow...
  echo
  echo -- Proudly served by a multi-line bash script V0.0.1
fi

