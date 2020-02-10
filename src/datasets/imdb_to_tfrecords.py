from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import makedirs
from os.path import join, exists
from glob import glob

import numpy as np

import tensorflow as tf
from shutil import copyfile

from .common import convert_to_example, ImageCoder

tf.app.flags.DEFINE_string('img_directory',
                           'datasets/human/imdb',
                           'image data directory')
tf.app.flags.DEFINE_string(
    'output_directory', '/home/rafal/dev/vd/hmr/tf_datasets/imdb/',
    'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 500,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 500,
                            'Number of shards in validation TFRecord files.')

FLAGS = tf.app.flags.FLAGS


def _add_to_tfrecord(image_path, label, coder, writer):
    with tf.gfile.FastGFile(image_path, 'rb') as f:
        image_data = f.read()

    image = coder.decode_jpeg(image_data)
    height, width = image.shape[:2]
    assert image.shape[2] == 3

    visible = label[2, :].astype(bool)

    min_pt = np.min(label[:2, visible], axis=1)
    max_pt = np.max(label[:2, visible], axis=1)
    center = (min_pt + max_pt) / 2.

    # kolejnosc punktow to samo co w mpii
    # filtrowanie

    pX = label[0]
    pY = label[1]
    pZ = label[2]

    # checks that each required point is present
    if (pX[0] == 0 or pX[1] == 0 or pX[2] == 0 or pX[3] == 0 or pX[4] == 0 or pX[5] == 0 or pX[8] == 0 or pX[9] == 0):
        return 0

    pA = abs(pY[8] - pY[2])
    pB = abs(pY[2] - pY[1])
    pC = abs(pY[1] - pY[0])
    pD = abs(pY[9] - pY[3])
    pE = abs(pY[3] - pY[4])
    pF = abs(pY[4] - pY[5])

    pG = abs(pX[8] - pX[2])
    pH = abs(pX[2] - pX[1])
    pI = abs(pX[1] - pX[0])
    pJ = abs(pX[9] - pX[3])
    pK = abs(pX[3] - pX[4])
    pL = abs(pX[4] - pX[5])

    pM = abs(pX[9] - pX[8])
    pN = abs(pX[2] - pX[3])
    pO = abs(pX[1] - pX[4])
    pP = abs(pX[0] - pX[5])

    perAB = pA / pB
    perBC = pB / pC
    perGA = pG / pA
    perHB = pH / pB
    perIC = pI / pC
    perJD = pJ / pD
    perKE = pK / pE
    perLF = pL / pF
    perAM = pA / pM
    perAN = pA / pN
    perAO = pA / pO
    perAP = pA / pP

    # checks that the person is in an standing position (better quality 0.35->0.2 | 1-1.8 -> 1-1.6)
    if (perGA > 0.55 or perHB > 0.55 or perIC > 0.55 or perJD > 0.55 or perKE > 0.55 or perLF > 0.55 or (
            perAB < 0.7 or perAB > 1.3) or (perBC < 1 or perBC > 1.8)):
        return 0

    # checks that the head is above the legs
    if (pY[0] < pY[8]):
        return 0

    # checks that the person is ahead (better quality 1.3-1.9 -> 1.5-1.7)
    if (perAM < 1.3 or perAM > 1.9):
        return 0

    # checks that the person is face to camera (reverse to change orientation)
    if (pX[8] > pX[9]):
        return 0

    # copyfile(image_path, 'datasets/human/lsp_dataset/standingpose/' + image_path[len(image_path)-10:len(image_path)])

    example = convert_to_example(image_data, image_path, height, width, label,
                                 center)

    writer.write(example.SerializeToString())

    return 1


def package(img_paths, labels, out_path, num_shards):
    """
    packages the images and labels into multiple tfrecords.
    """
    coder = ImageCoder()

    i = 0
    fidx = 0
    while i < len(img_paths):
        # Open new TFRecord file.
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(img_paths) and j < num_shards:
                if i % 100 == 0:
                    print('Converting image %d/%d' % (i, len(img_paths)))
                j += _add_to_tfrecord(img_paths[i],
                                      labels[:, :, i],
                                      coder,
                                      writer)
                i += 1

        fidx += 1


def load_mat(fname):
    import scipy.io as sio
    res = sio.loadmat(fname)
    # this is 3 x 14 x 2000
    return res['joints']

def map_to_lsp_joints(labels):
    _COMMON_JOINT_IDS = [
        10,  # R ankle
        9,  # R knee
        8,  # R hip
        11,  # L hip
        12,  # L knee
        13,  # L ankle
        4,  # R Wrist
        3,  # R Elbow
        2,  # R shoulder
        5,  # L shoulder
        6,  # L Elbow
        7,  # L Wrist
        1,  # Neck top
        0,  # Head top
    ]

    num_images = labels.shape[2]
    mapped_labels = np.zeros((3, 14, num_images))

    for i, jid in enumerate(_COMMON_JOINT_IDS):
        mapped_labels[:, i, :] = labels[:, jid, :]

    return mapped_labels

def process_imdb(img_dir, out_dir, num_shards_train, num_shards_test):
    """
    Args:
      img_dir: string, root path to the data set.
      num_shards: integer number of shards for this data set.
    """
    # Load labels 3 x 14 x N
    labels = load_mat(join(img_dir, 'result.mat'))
    if labels.shape[0] != 3:
        labels = np.transpose(labels, (1, 0, 2))

    labels = map_to_lsp_joints(labels)

    all_images = sorted([f for f in glob(join(img_dir, 'images/*.jpg'))])

    train_out = join(out_dir, 'train_%03d.tfrecord')
    package(all_images, labels, train_out, num_shards_train)


def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)
    process_imdb(FLAGS.img_directory, FLAGS.output_directory,
                 FLAGS.train_shards, FLAGS.validation_shards)


if __name__ == '__main__':
    tf.compat.v1.app.run()
