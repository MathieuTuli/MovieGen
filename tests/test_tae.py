from train_moviegen import TAE, TAEConfig


def test_tae_weight_loading():
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")


def test_tae_encoder():
    # single frame video
    import torch
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
    x = torch.zeros((3, 1, 3, 32, 32))  # [b, seq, c, w, h]
    x = tae.encode(x)


def test_tae_decoder():
    import torch
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
    x = torch.zeros((3, 1, 16, 32, 32))  # [b, seq, c, w, h]
    tae.decode(x)
    x = torch.zeros((3, 32, 16, 32, 32))  # [b, seq, c, w, h]
    tae.decode(x)
