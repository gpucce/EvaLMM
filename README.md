# EvaLMM

This repository can be used to split a video into subvideos and then generated captions for subvideos by first captionin the frames and then joining the captions together.

The video we used are taken from https://huggingface.co/datasets/shawshankvkt/Walking_Tours . This are egocentric videos of a person walking around a city.

## Video Splitting

The command `python evalmm/video_splitting/split_video.py video_location output_location` will split one of the videos from the above collection into a list of images, taken 1 every 7 seconds.

## Video Captioning

The `scripts` folder contains the command to first caption all the frames, that command can be run through 

`bash scripts/run_descriptive_captioning.sh` make sure to adjust the path to what you used as `output_path` in the above command

Finally to obtain a joint caption for each subvideo one can run the command

`bash scripts/run_llama_captioning.sh`

This last command will also use ffmpeg (which should be installed) to create the short videos with the caption, this requires quite some time.

> Note that both the scripts in the `scripts` folder assune that one has access to 4 different GPUs but it can be easily changed.
