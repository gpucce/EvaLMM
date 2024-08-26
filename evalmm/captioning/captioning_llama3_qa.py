from pathlib import Path
import json
import os
import random
from copy import deepcopy
import torch
from itertools import islice
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

CITY_FPS = {
    "Bangkok": 29.97,
    "Chiang_Mai": 30,
    "Kuala_Lumpur": 30,
    "Amsterdam": 30,
    "Venice": 29.97,
    "Wildlife": 29.97,
    "Istanbul": 30,
    "Singapore": 30,
    "Stockholm": 30,
    "Zurich": 30,
}

random.seed(41)

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

def _generate_by_messages(model, tokenizer, messages, output_only=True):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # response = outputs[0][input_ids.shape[-1]:]
    # if not output_only:
    #     response = outputs[0]

    return {
        "text": tokenizer.decode(outputs[0], skip_special_tokens=True),
        "response": tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    }

def generate_all_at_once(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    output = _generate_by_messages(model, tokenizer, messages)
    new_message = output["text"]
    questions = output["response"]

    messages.append({"role": "assistant", "content": questions})
    messages.append({"role": "user", "content": FINAL_PROMPT})

    output = _generate_by_messages(model, tokenizer, messages, output_only=False)

    full_text = output["text"]
    answers = output["response"]

    return {"questions": questions, "answers": answers, "full_text": full_text}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_img.jpg")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--city", type=str, required=True, nargs="+")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/videos/")
    parser.add_argument("--avoid_extract_video", action="store_true", default=False)
    return parser.parse_args()

def get_frame_number_from_path(frame_path):
    return int(frame_path.split("_")[1].replace(".jpg", ""))

def main(args):
    cities = args.city
    data_path = Path(args.data_path)
    output_path = Path(args.output_dir)
    max_videos = args.max_videos

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


    captions = {}
    batch_size = args.batch_size
    id = 0
    for city in cities:
        with open(data_path / "videos_images_llava_captions" / city / "captions.json") as jf:
            descriptions = json.load(jf)
        frames = list(descriptions.keys())
        sorted_frames = sorted(frames, key=get_frame_number_from_path)
        fps = CITY_FPS[city]
        random_starts = random.sample(
            range(len(sorted_frames)), args.max_videos)
        video_path = data_True
        for start_frame_name_idx in random_starts:
            if max_videos > 0 and id >= max_videos:
                break
            id += 1

            start_frame_name = sorted_frames[start_frame_name_idx]
            start_frame_path_name = (start_frame_name + ".jpg"
                if not start_frame_name.endswith(".jpg") else start_frame_name)
            start_frame_number = get_frame_number_from_path(start_frame_path_name)
            start_frame_path = data_path / "full_walking_tours_images" / f"{city}_imgs" / start_frame_path_name

            n_steps_forward = random.randint(4, 7)
            final_frame_name_idx = start_frame_name_idx + n_steps_forward
            try:
                final_frame_name = sorted_frames[final_frame_name_idx]
            except:
                id = 0
                break
            final_frame_path_name = (final_frame_name + ".jpg"
                if not final_frame_name.endswith(".jpg") else final_frame_name)
            final_frame_number = get_frame_number_from_path(final_frame_path_name)
            final_frame_path = data_path / "full_walking_tours_images" / f"{city}_imgs" / final_frame_path_name

            starting_second = start_frame_number / fps
            if starting_second > 1:
                starting_second -= 1 # add some seconds to the start
            lasting_time = (final_frame_number - start_frame_number) / fps
            lasting_time += 1 # add some seconds to the end

            _output_path = Path(output_path / city)

            video_output_path = _output_path / "videos"
            video_output_path.mkdir(parents=True, exist_ok=True)
            video_output_path = (str(video_output_path / f"{start_frame_name}_video") + ".mp4").replace(".jpg", "")
            qa_output_path = _output_path / "qa"
            qa_output_path.mkdir(parents=True, exist_ok=True)
            qa_output_path = (str(qa_output_path / f"{start_frame_name}_qa") + ".json").replace(".jpg", "")

            if not args.avoid_extract_video:
                # os.system(f"ffmpeg -y -i {video_path} -ss {starting_second} "
                #               f"-t {lasting_time} -c copy {video_output_path}")
                os.system(f"ffmpeg -y -i {video_path} -ss {starting_second} "
                            f"-t {lasting_time} -c:v libx264 -crf 23 -c:a aac {video_output_path}")

            all_frames_descriptions = [
                descriptions[frame][0]["a"] for frame in
                sorted_frames[start_frame_name_idx:final_frame_name_idx]]

            prompt = deepcopy(PROMPT)
            for idx, desc in enumerate(all_frames_descriptions):
                prompt = prompt.format(infos=f"Description of frame number {idx}:\n" + desc + "\n\n{infos}")
                prompt.format(infos="")

            generated = generate_all_at_once(model, tokenizer, prompt)
            with open(qa_output_path, "w") as jf:
                json.dump(generated, jf)
        id = 0


if __name__ == "__main__":
    args = parse_args()

    SYSTEM_PROMPT = "You are an acute observer keen on asking tricky and precise questions about videos."
    PROMPT = "Given the following descriptions about events in a video ask 5 true/false questions investigating the relative occurrence in time of two of these events. In the questions describe the events.\n\nEvents:\n\n{infos}\n\nPlease never mention the words frames, photos, scenes, event and Event and only write the questions."
    FINAL_PROMPT = "Now ask the same questions as above but reversing the order of the events."

    print(args)
    main(args)