from pathlib import Path
import json
from tqdm.auto import tqdm
import torch
from itertools import islice
from argparse import ArgumentParser
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

def get_frame_number_from_path(frame_path):
    return int(frame_path.stem.split("_")[1])

def get_images(args):
    imgs = list(Path(args.image_path).glob("frame_*"))
    imgs = sorted(imgs, key=get_frame_number_from_path)
    if args.max_images:
        imgs = imgs[:args.max_images]
    return imgs

def get_prompt(text):
    return PROMPT_TEMPLATE.format(text=text)

def caption_image(args, image_paths, model, processor):
    model.to(args.device)
    images = [Image.open(image) for image in image_paths]
    answers = {image.name: [] for image in image_paths}
    for idx, _prompt in enumerate(PROMPTS):
        if idx == 0:
            prompts = [get_prompt(_prompt)] * len(images)
        else:
            # prompt = get_prompt(answers[0]["a"] + "\n" + _prompt)
            prompts = [get_prompt(_answer[0]["a"] + "\n" + _prompt) for _, _answer in answers.items()]

        processor.tokenizer.padding_side = "left"
        inputs = processor(text=prompts, images=images,
                           return_tensors="pt", padding=True).to(args.device)
        generate_ids = model.generate(**inputs, max_length=1024, min_new_tokens=20)
        _answers = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # [:len(images)]

        for image_path, _answer in zip(image_paths, _answers):
            answers[str(image_path.name)].append({"q": _prompt, "a": _answer.split("[/INST]")[1]})

        # answers.append({"q":_prompt, "a":_answer.split("[/INST]")[1]})
        # print(f"{idx} ANSWER ######################################", answers)
        # print(answers)

    return answers

def main(args):
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "./llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True)

    processor = AutoProcessor.from_pretrained("./llava-v1.6-mistral-7b-hf")

    images = get_images(args)
    print(f"N. Images {len(images)}")
    print(args.image_path)
    captions = {}
    batch_size = args.batch_size
    progressbar = tqdm(range(len(images)))
    for image in batched(images, batch_size):
        captions = {**captions, **caption_image(args, image, model, processor)}
        progressbar.update(batch_size)

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "captions.json", "w") as f:
        json.dump(captions, f)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--captioning_type", type=str, choices=["qa", "description"])
    parser.add_argument("--image_path", type=str, default="test_img.jpg")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    PROMPT_TEMPLATE = "[INST] <image>\n{text} [/INST]"
    if args.captioning_type == "qa":
        PROMPTS = [
            "Can you ask 5 questions about this photo?",
            "Please answer the previous 5 questions about this photo:",
            f"Please answer the previous 5 questions about this photo knowing that it was taken in {args.city}:",
        ]
    elif args.captioning_type == "description":
        PROMPTS = [
            "Write a list of extensive and detailed descriptions of at most three elements in this photo choosing among people, objects, vehicles, signs, writings, bars, restaurants or others. "
            "Describe only one of each kind, focus on remarkable things and avoid those that are colored black or white. "
            "Describe them one by one, for each of them give as many details as possible in a few sentences, such as colors, shapes, activities and more, so it can't be mistaken for another one. "
            "If there are writings on walls or signs report them as well. Add geometric information if possible, such as sign shape and similar."
            "Avoid talking about the sky or the weather in general, don't mention signs that are upside down and don't mention the color of the sky. "
            "Do not talk about reflections or left and right and never mention foreground or background and never use the word remarkable."
        ]

    print(PROMPTS)
    main(args)