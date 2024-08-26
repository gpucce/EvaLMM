#!/bin/bash

cities=(Amsterdam Bangkok Chiang_Mai Istanbul Kuala_Lampur Singapore Stockholm Venice Zurich)

N=4
(
for city in ${cities[@]}
do
    ((i=i%N)); ((i++==0)) && wait
    echo "${i} ${city}"
    echo $city
    img_path="/home/gpucce/data/evalmm/walking_tours_images/${city}_imgs/"
    echo $img_path
    python captioning_llava.py \
      --captioning_type qa \
      --image_path $img_path \
      --output_dir "outputs/qa/$city" \
      --city $city \
      --batch_size 16 \
      --device $(($i - 1)) &
done
)
