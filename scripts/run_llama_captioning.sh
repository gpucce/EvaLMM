
cities=(Amsterdam Bangkok Chiang_Mai Istanbul Kuala_Lumpur Singapore Stockholm Venice Zurich)
# cities=(Amsterdam Bangkok)

N=4
(
for city in ${cities[@]}
do
    ((i=i%N)); ((i++==0)) && wait
    echo "${i} ${city}"
    echo $city
    img_path="/home/gpucce/data/evalmm/data/full_walking_tours_images/${city}_imgs/"
    echo $img_path
    python evalmm/captioning/captioning_llama3_descriptive.py \
      --data_path /home/gpucce/data/evalmm/data/videos_images_llava_descriptive_captions_v5 \
      --output_dir "/home/gpucce/data/evalmm/data/videos_captions_descriptive_v5/" \
      --city $city \
      --batch_size 16 \
      --avoid_extract_video \
      --device $(($i - 1)) &
done
)