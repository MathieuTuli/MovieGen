from tae import TAE, TAEConfig


def test_tae_weight_loading():
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.from_pretrained("pretrained-weights/tae/model.ckpt",
                        ignore_keys=[
                            "encoder.conv_out",
                            "decoder.conv_in",
                            "quant_conv",
                            "post_quant_conv",
                            ])


def test_tae_encoder():
    # single frame video
    import torch
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.to(device)
    x = torch.zeros((3, 1, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    x = tae.encode(x)
    B, T, C, H, W = x.sample().shape
    assert C == 16
    assert T == 1
    assert H == 4
    assert W == 4
    x = torch.zeros((3, 128, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    x = tae.encode(x)
    B, T, C, H, W = x.sample().shape
    assert C == 16
    assert T == 16
    assert H == 4
    assert W == 4
    x = torch.zeros((3, 33, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    x = tae.encode(x)
    B, T, C, H, W = x.sample().shape
    assert C == 16
    assert T == 5  # rounded up to avoid frame loss
    assert H == 4
    assert W == 4
    del tae, x


def test_tae_decoder():
    import torch
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.to(device)
    x = torch.zeros((3, 1, 16, 4, 4), device=device)  # [b, seq, c, w, h]
    y = tae.decode(x)
    assert y.shape[1] == 8
    x = torch.zeros((3, 3, 16, 3, 3), device=device)  # [b, seq, c, w, h]
    y = tae.decode(x)
    assert y.shape[1] == 3 * 8
    del tae, x


def test_tae_forward():
    import torch
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tae_config = TAEConfig()
    tae = TAE(tae_config)
    tae.to(device)
    x = torch.zeros((3, 1, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    dec, post, _ = tae(x, "val", 0, 0)
    assert dec.shape == x.shape
    x = torch.zeros((3, 11, 3, 32, 32), device=device)  # [b, seq, c, w, h]
    dec, post, _ = tae(x, "val", 0, 0)
    assert dec.shape == x.shape
    del tae, x
