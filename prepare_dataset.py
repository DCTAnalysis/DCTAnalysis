"""Prepare image data."""
import argparse
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.dataset import image_paths, serialize_data
from src.image_np import dct2, load_image
from src.math import welford

# FFHQ experiments
# TRAIN_SIZE = 10_000
# VAL_SIZE = 1_000
# TEST_SIZE = 5_000

# LSUN/CelebA experiments
TRAIN_SIZE = 100_000
VAL_SIZE = 10_000
TEST_SIZE = 30_000


def _collect_image_paths(dirictory):
    images = image_paths(dirictory)
    assert len(images) >= TRAIN_SIZE + VAL_SIZE + \
        TEST_SIZE, f"{len(images)} - {dirictory}"

    train_dataset = images[:TRAIN_SIZE]
    val_dataset = images[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]
    test_dataset = images[TRAIN_SIZE +
                          VAL_SIZE: TRAIN_SIZE + VAL_SIZE + TEST_SIZE]
    assert len(
        train_dataset) == TRAIN_SIZE, f"{len(train_dataset)} - {dirictory}"

    assert len(val_dataset) == VAL_SIZE, f"{len(val_dataset)} - {dirictory}"

    assert len(test_dataset) == TEST_SIZE, f"{len(test_dataset)} - {dirictory}"

    return (train_dataset, val_dataset, test_dataset)


def collect_all_paths(dirs):
    directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(dirs).iterdir())))

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for i, directory in enumerate(directories):
        train, val, test = _collect_image_paths(directory)

        train = zip(train, [i] * len(train))
        val = zip(val, [i] * len(val))
        test = zip(test, [i] * len(test))

        train_dataset.extend(train)
        val_dataset.extend(val)
        test_dataset.extend(test)

        del train, val, test

    train_dataset = np.asarray(train_dataset)
    val_dataset = np.asarray(val_dataset)
    test_dataset = np.asarray(test_dataset)

    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    return train_dataset, val_dataset, test_dataset


def convert_images(inputs, tf=False, grayscale=True, normalize=False):
    image, label = inputs
    image = load_image(image, grayscale=grayscale, tf=tf)
    image = image.astype(np.float32)
    if normalize:
        image /= 127.5
        image -= 1.

    return (image, label)


def convert_images_dct(inputs, tf=False, grayscale=True, log_scaled=False, abs=False):
    image, label = inputs
    image = load_image(image, grayscale=grayscale, tf=tf)
    image = dct2(image)
    if log_scaled:
        image = np.abs(image)
        image += 1e-12
        image = np.log(image)

    if abs:
        image = np.abs(image)

    return (image, label)


def create_directory_np(output_path, images, convert_function):
    os.makedirs(output_path, exist_ok=True)
    converted_images = map(convert_function, images)

    labels = []
    for i, (img, label) in enumerate(converted_images):
        with open(f"{output_path}/{i:06}.npy", "wb+") as f:
            np.save(f, img)

        labels.append(label)

    with open(f"{output_path}/labels.npy", "wb+") as f:
        np.save(f, labels)


def normal_mode(directory, encode_function, outpath):
    (train_dataset, val_dataset, test_dataset) = collect_all_paths(directory)
    create_directory_np(f"{outpath}_train",
                        train_dataset, encode_function)
    create_directory_np(f"{outpath}_val",
                        val_dataset, encode_function)
    create_directory_np(f"{outpath}_test",
                        test_dataset, encode_function)


def create_directory_tf(output_path, images, convert_function):
    os.makedirs(output_path, exist_ok=True)

    converted_images = map(convert_function, images)
    converted_images = map(serialize_data, converted_images)

    def gen():
        for serialized in converted_images:
            yield serialized

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=tf.string, output_shapes=())
    filename = f"{output_path}/data.tfrecords"
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)


def tfmode(directory, encode_function, outpath):
    train_dataset, val_dataset, test_dataset = collect_all_paths(directory)
    create_directory_tf(f"{outpath}_train_tf",
                        train_dataset, encode_function)
    create_directory_tf(f"{outpath}_val_tf",
                        val_dataset, encode_function)
    create_directory_tf(f"{outpath}_test_tf",
                        test_dataset, encode_function)


def normalize(array, mean, std, encode_function):
    image, label = encode_function(array)
    image = (image - mean) / std
    return image, label


def main(args):
    encode_function = convert_images_dct
    output = f"{args.DIRECTORY.rstrip('/')}"

    if args.raw:
        encode_function = functools.partial(
            convert_images, normalize=args.normalize)
        output += "_raw"
        if args.normalize:
            output += "_prnu"

    if args.color:
        encode_function = functools.partial(encode_function, grayscale=False)
        output += "_color"

    if args.log:
        encode_function = functools.partial(encode_function, log_scaled=True)
        output += "_log_scaled"

    if args.abs:
        encode_function = functools.partial(encode_function, abs=True)
        output += "_abs"

    if args.mode == "tfrecords":
        encode_function = functools.partial(encode_function, tf=True)

    if args.normalize:
        train, _, _ = collect_all_paths(args.DIRECTORY)
        images = map(encode_function, train)

        images = map(lambda x: x[0], images)
        mean, var = welford(images)
        std = np.sqrt(var)
        output += "_normalized"
        encode_function = functools.partial(
            normalize, encode_function=encode_function, mean=mean, std=std)

    if args.mode == "normal":
        normal_mode(args.DIRECTORY, encode_function, output)
    elif args.mode == "tfrecords":
        tfmode(args.DIRECTORY, encode_function, output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Directory to convert.",
                        type=str)

    parser.add_argument("--raw", "-r", help="Save image data as raw image.",
                        action="store_true")
    parser.add_argument("--log", "-l", help="Log scale Images.",
                        action="store_true")
    parser.add_argument("--abs", "-a", help="Absolute Images.",
                        action="store_true")

    parser.add_argument("--color", "-c", help="Compute as color instead.",
                        action="store_true")
    parser.add_argument("--normalize", "-n", help="Normalize data.",
                        action="store_true")

    modes = parser.add_subparsers(
        help="Select the mode {normal|tfrecords}", dest="mode")

    _ = modes.add_parser("normal")
    _ = modes.add_parser("tfrecords")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
