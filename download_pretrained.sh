#!/bin/bash
mkdir pretrained-weights
mkdir pretrained-weights/tae
mkdir pretrained-weights/ul2
mkdir pretrained-weights/byt5
mkdir pretrained-weights/metaclip

# TAE
cd pretrained-weights/tae
if [ ! -f model.ckpt ]; then
    wget https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
    unzip kl-f8.zip
else
    echo "tae kl-f8 model.ckpt exists, skipping..." 
fi
cd ../..

# for ul2 I use hugginface


# byt5
#- cd pretrained-weights/byt5
#- wget ???
#- unzip ???.zip
#- cd ../..

# metaclip
cd pretrained-weights/metaclip
if [ ! -f G14_fullcc2.5b.pt ]; then
    wget https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt
else
    echo "metaclip G14_fullcc2.5b.pt exists, skipping..." 
fi
cd ../..
