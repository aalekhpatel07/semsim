#!/bin/sh

random_filename () {
    echo "$(echo $RANDOM | md5sum | head -c 10).jpg"
}

dog_api_url="https://dog.ceo/api/breed/$1/images/random"

mkdir -p images/dog_api/$1

for x in $(seq 1 $2)
do
    filename="$(random_filename)"
    curl $dog_api_url | fx 'this.message' | xargs curl -o ./images/dog_api/$1/$filename &
done
