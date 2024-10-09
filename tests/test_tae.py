from train_moviegen import TAE, TAEConfig


def test_tae_weight_loading():
    tae_config = TAEConfig()
    tae = TAE(tae_config)

    tae.from_pretrained("pretrained-weights/tae/model.ckpt")
