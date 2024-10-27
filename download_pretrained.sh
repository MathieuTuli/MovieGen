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
python -c "from transformers import T5EncoderModel; T5EncoderModel.from_pretrained('google/ul2')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/ul2')"

# for byt5 I use hugginface
python -c "from transformers import T5EncoderModel; T5EncoderModel.from_pretrained('google/byt5-small')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/byt5-small')"

# metaclip
cd pretrained-weights/metaclip
if [ ! -f G14_fullcc2.5b.pt ]; then
    wget https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt
else
    echo "metaclip G14_fullcc2.5b.pt exists, skipping..." 
fi
if [ ! -f bpe_simple_vocab_16e6.txt.gz ]; then
    wget https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz
else
    echo "metaclip bpe_simple_vocab_16e6.txt.gz exists, skipping..." 
fi
cd ../..
