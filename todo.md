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

# TAE
- [x] validate training working
- [ ] batch multi-frame masking for loss
- [ ] build inference pipeline
- [x] validate outlier penalty loss which handles spots
    - it produces odd results to be honest, introducing even more artifacts for me, needs more investigation
- [ ] resume training
    - ~save opt state~
    - save loss state
    - log state
- [ ] improve logging
    - right now it just dumps to a file, which is fine just clean the dump

# MovieGen
- [ ] build training pipeline
- [ ] build inference pipeline

# Optimizations
- [ ] my dtypes are just plopped in and not considered, revisit
- [ ] `torch.compile` is just plopped from llm.c also, revisit
- [ ] Model sharding: implement techniques from paper
- [ ] TAE: DDP loader
- [ ] MovieGen: DDP loader
- [ ] TAE: efficient inference using temporal tiling

# QOL
- [ ] implement terminal-based logging/chart plotting
