import os
import os.path as osp
import argparse
from einops import rearrange
from tqdm import tqdm

import torch
from torchvision.utils import make_grid, save_image

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import DDIMScheduler
from pipeline_tvgen import TVGenPipeline
from models.unet3d import UNet3DConditionModel


def parse_args():
	parser = argparse.ArgumentParser(description="Simple example of text->video generation")

	parser.add_argument(
		"--pretrained_model_path",
		type=str,
		default="CompVis/stable-diffusion-v1-4",
		help="Path to pretrained model or model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--gpu_id",
		type=int,
		default=0,
		help="GPU index, start from 0. Negative value uses cpu",
	)
	parser.add_argument(
		"--prompts",
		type=str,
		default=None,
		nargs="+",
		required=True,
		help="A set of prompts to generate videos.",
	)
	parser.add_argument(
		"--video_length",
		type=int,
		default=8,
		help="Number of frames in generated videos.",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=512,
		help="Frame height.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=512,
		help="Frame width.",
	)
	parser.add_argument(
		"--num_videos_per_prompt",
		type=int,
		default=1,
		help="Number of generated videos for each input prompt.",
	)
	parser.add_argument(
		"--guidance_scales",
		type=float,
		default=7.5,
		nargs="+",
		help="Guidance scale for classfier-free guidance inference.",
	)
	parser.add_argument(
		"--num_inference_steps",
		type=int,
		default=50,
		help="Number for inference steps of diffusion scheduler.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./outputs",
		help="Directory to save generated videos.",
	)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	# in case of cuda oom errors, decrease width, height, video_length, num_videos_per_prompt
	# too small resolution may lead to low image quality
	
	cfg = parse_args()
	
	if cfg.gpu_id >= 0:
		device = f"cuda:{cfg.gpu_id}"
	else:
		device = "cpu"
	os.makedirs(cfg.output_dir, exist_ok=True)
	
	vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae")
	tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
	unet = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_model_path, subfolder="unet")
	text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder")
	scheduler = DDIMScheduler.from_pretrained(cfg.pretrained_model_path, subfolder="scheduler")
	
	tvgen_pipeline = TVGenPipeline(
		vae=vae,
		tokenizer=tokenizer,
		text_encoder=text_encoder,
		unet=unet,
		scheduler=scheduler
	)
	tvgen_pipeline.to(device)
	# generate videos for each prompt one by one to save gpu memory
	for prompt in tqdm(cfg.prompts):
		# generated video in shape (b c f h w)
		# b = n_guidance_scales X n_videos_per_prompt
		video = tvgen_pipeline(prompt=prompt,
		                       video_length=cfg.video_length,
		                       height=cfg.height,
		                       width=cfg.width,
		                       num_inference_steps=cfg.num_inference_steps,
		                       guidance_scales=cfg.guidance_scales,
		                       num_videos_per_prompt=cfg.num_videos_per_prompt,
		                       progress=False,
		                       return_dict=False).cpu()
		# ToDo: save video and visualize
		video = rearrange(video, "(g n) c f h w -> g (n f) c h w", g=len(cfg.guidance_scales))
		for gi in range(len(cfg.guidance_scales)):
			save_image(make_grid(video[gi], nrow=cfg.video_length, normalize=True, value_range=(0, 1)),
			           f"{cfg.output_dir}/{prompt}_gs={cfg.guidance_scales[gi]}.png")



