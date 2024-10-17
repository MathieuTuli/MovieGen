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
    x = torch.zeros((3, 32, 3, 32, 32))  # [b, seq, c, w, h]
    x = tae.encode(x)
    del tae, x


def test_tae_decoder():
    import torch
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
    x = torch.zeros((3, 1, 16, 4, 4))  # [b, seq, c, w, h]
    tae.decode(x)
    x = torch.zeros((3, 32, 16, 4, 4))  # [b, seq, c, w, h]
    tae.decode(x)
    del tae, x


def test_tae_forward():
    import torch
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
    x = torch.zeros((3, 1, 3, 32, 32))  # [b, seq, c, w, h]
    dec, post = tae(x)
    assert dec.shape == x.shape
    x = torch.zeros((3, 32, 3, 32, 32))  # [b, seq, c, w, h]
    dec, post = tae(x)
    assert dec.shape == x.shape
    del tae, x
