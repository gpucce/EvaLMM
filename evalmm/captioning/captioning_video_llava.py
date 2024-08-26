
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import json, re
import av
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def main(args):
    cities = args.city

    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    # video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")

    data_path = Path(args.data_path)


    id = 0
    for city in cities:
        _data_path = data_path / "videos_captions_descriptive" / f"{city}"
        videos = sorted(str(i.resolve()) for i in (_data_path / "videos").iterdir())
        qas = sorted(str(i.resolve()) for i in (_data_path / "qa").iterdir())
        for video_path, qa_path in zip(videos, qas):
            if id >= args.max_videos:
                return
            id += 1

            with open(qa_path) as f:
                captions = json.load(f)

            questions = re.split("\d\.", captions["questions"])
            questions = [q.replace("\n", "") for q in questions][1:]
            questions = [f"{idx + 1}. " + q for idx, q in enumerate(questions)]

            container = av.open(video_path)

            # sample uniformly 8 frames from the video
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = read_video_pyav(container, indices)

            print(video_path)
            for question in questions:
                # prompt = "USER: <video>Why is this video funny? ASSISTANT:"
                prompt = f"USER: <video>{question} ASSISTANT:"

                # Tokenize
                inputs = processor(text=prompt, videos=clip, return_tensors="pt")

                # Generate
                generate_ids = model.generate(**inputs, max_length=80)
                print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_img.jpg")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--city", type=str, required=True, nargs="+")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/videos/")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())