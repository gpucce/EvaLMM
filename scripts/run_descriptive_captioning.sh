#!/bin/bash

cities=(Amsterdam Bangkok Chiang_Mai Istanbul Singapore Stockholm Venice Zurich)
# cities=(Chiang_Mai Istanbul)

N=4
(
for city in ${cities[@]}
do
    ((i=i%N)); ((i++==0)) && wait
    echo "${i} ${city}"
    echo $city
    img_path="/home/gpucce/data/evalmm/data/full_walking_tours_images/${city}_imgs"
    echo $img_path
    python evalmm/captioning/captioning_llava.py \
      --captioning_type description \
      --image_path $img_path \
      --output_dir "/home/gpucce/data/evalmm/data/videos_images_llava_descriptive_captions_v5/$city" \
      --city $city \
      --batch_size 8 \
      --device $(($i - 1)) &

done
)
