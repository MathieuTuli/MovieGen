---
TODOs for MovieGen implementation
- I'm working on this on the side of many other threads so I surely will need a reminder of what I was doing as I tackle this in week-chunks
---

# General
- [ ] fps considerations?
    - tbh I forget what this means
    - wait, from paper:
    - *"Controlling the FPS. We use FPS conditioning to control the length of the generated videos by pre-appending the sampling FPS value of each training video to the input text prompt (e.g., “FPS-16”). During pre-training, we sample video clips at their original FPS with minimum of 16 FPS. In finetuning, we sample clips at two fixed FPS values of 16 and 24."*

# Text Encoders
- [x] copy baselines
- [x] weight loading
- [x] implement the basic map/concat pipeline they use
- [ ] fine tune MetaCLIP on longer text captions  (77 to 256 context length)
- [ ] character-level ByT5 encoder is only used to encode visual text, i.e., the part of the text prompt that may explicitly ask for a character string to be generated in the output.

# TAE
- [x] validate training working
- [x] batch multi-frame masking for loss
- [ ] build inference pipeline
- [x] validate outlier penalty loss which handles spots
    - it produces odd results to be honest, introducing even more artifacts for me, needs more investigation
    - it's working, but the weight needs to be tuned different from the paper
- [x] resume training
    - ~save opt state~
    - save loss state
    - log state
- [ ] improve logging
    - right now it just dumps to a file, which is fine just clean the dump

# MovieGen
- [ ] build training pipeline
- [ ] build inference pipeline
- [ ] controlling fps: preprend "FPS-16"
    - pre-training, use default fps, prepended in prompt
    - in fine-tuning, resample to 16 or 24
- [ ] Bias is honestly randomly set to true or false in many places
    - e.g. conv in patchifier
- [ ] arbitrary resolutions/aspect ratios : currently forcing 256x256
    - Currently, the patchifier/flattening assumes 256x256 inputs, with padded/masked frames
    - For arbitrary resolutions/aspect ratios/frames, the code will need to be updated
- [ ] token dropout?

# Optimizations
- [ ] my dtypes are just plopped in and not considered, revisit
- [ ] `torch.compile` is just plopped from llm.c also, revisit
- [ ] Model sharding: implement techniques from paper
- [ ] TAE: DDP loader
- [ ] MovieGen: DDP loader
- [ ] TAE: efficient inference using temporal tiling
- [ ] TAE: dataloader loads and discards a video file, could optimize this and use all frames in a video file or something
- [ ] TAE: grad accum
- [ ] TAE: optimize multi-processing dataloader
- [ ] TAE: valset is split from DDP, which shouldn't happen
- [ ] TAE: preprocess dataset and preload chunks like in llm.c

# QOL
- [ ] implement terminal-based logging/chart plotting
