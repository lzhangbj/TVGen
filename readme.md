## Installation
```angular2html
conda env create -f tvgen.yml
conda activate tvgen
```

## Inference
```angular2html
python main.py \
--pretrained_model_path CompVis/stable-diffusion-v1-4 \
--gpu_id 0 \
--prompts fire explosion water-burbling \
--video_length 4 \
--height 320 \
--width 512 \
--num_videos_per_prompt 0 \
--guidance_scales 3.0, 7.5 \
--output_dir ./outputs
```

It cost lots of GPU memory for generation. 
In case of cuda oom errors, 
decrease width, height, video_length, num_videos_per_prompt.
Too small resolution may lead to low image quality