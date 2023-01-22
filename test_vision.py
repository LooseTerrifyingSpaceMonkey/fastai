from fastai.vision.all import *
from pathlib import *


def test_basic_vision_pets():
    path = untar_data(URLs.PETS)
    files = get_image_files(path / "images")

    def label_func(f): return f[0].isupper()

    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

    dls.show_batch()
