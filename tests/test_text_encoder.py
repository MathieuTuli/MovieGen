from text_encoder import TextEncoder, TextEncoderConfig


def test_forward():
    te = TextEncoder(TextEncoderConfig(models={"ul2"}))
    del te
    te = TextEncoder(TextEncoderConfig(models={"byt5"}))
    del te
