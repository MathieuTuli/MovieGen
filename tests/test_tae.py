from train_moviegen import TAE, TAEConfig


def test_tae_weight_loading():
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")


def test_tae_encoder():
    # single frame video
    import torch
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.to(device)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
    x = torch.zeros((3, 1, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    x = tae.encode(x)
    assert x.sample().shape[1] == 1
    x = torch.zeros((3, 128, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    x = tae.encode(x)
    assert x.sample().shape[1] == 16 # 128 // 8
    x = torch.zeros((3, 33, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    x = tae.encode(x)
    assert x.sample().shape[1] == 4  # 33 // 8
    del tae, x


def test_tae_decoder():
    import torch
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.to(device)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
    x = torch.zeros((3, 1, 16, 4, 4), device=device)  # [b, seq, c, w, h]
    tae.decode(x)
    x = torch.zeros((3, 3, 16, 3, 3), device=device)  # [b, seq, c, w, h]
    tae.decode(x)
    del tae, x


def test_tae_forward():
    import torch
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.to(device)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
    x = torch.zeros((3, 1, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    dec, post = tae(x)
    assert dec.shape == x.shape
    x = torch.zeros((3, 11, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    dec, post = tae(x)
    assert dec.shape == x.shape
    del tae, x
