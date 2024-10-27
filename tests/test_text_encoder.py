from text_encoder import TextEncoder, TextEncoderConfig


def test_init():
    te = TextEncoder(TextEncoderConfig(models={"ul2"}))
    del te
    te = TextEncoder(TextEncoderConfig(models={"byt5"}))
    del te
    te = TextEncoder(TextEncoderConfig(models={"ul2", "byt5"}))
    del te
    te = TextEncoder(TextEncoderConfig(models={"ul2", "byt5", "metaclip"}))
    del te


def test_forward_byt5():
    te = TextEncoder(TextEncoderConfig(models={"byt5"}))
    te.cuda()
    inputs = te.tokenize("the world is a wonderous place", "cuda")
    te(inputs)
    inputs = te.tokenize("NULL", "cuda")
    te(inputs)


def test_forward_all_cpu():
    te = TextEncoder(TextEncoderConfig(models={"byt5", "ul2", "metaclip"}))
    inputs = te.tokenize(["the world is a wonderous place"])
    te(inputs)
