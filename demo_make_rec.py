import sys
sys.path.insert(0, '/opt/hdfs/user/he.huang/mxnet-python-binds/mxnet-zongbo')
import mxnet as mx
import os
import imageio
import numpy as np
import cv2
import random


def read_list(imglst_path):
    img_list = {}
    with open(imglst_path) as fin:
        for line in iter(fin.readline, ''):
            line = line.strip().split('\t')
            img_list[line[-1]] = int(line[0])
    return img_list


def make_rec_from_img(img_dir, save_rec_prefix):
    lst_fn = open(save_rec_prefix + '.lst', 'w')
    imgrec = mx.recordio.MXIndexedRecordIO(save_rec_prefix + '.idx', save_rec_prefix + '.rec', 'w')

    img_list = os.listdir(img_dir)
    # corrupt = os.listdir('corrupt')
    # img_list = list(set(img_list) - set(corrupt))
    random.seed(100)
    random.shuffle(img_list)
    num_images = len(img_list)

    for i in range(num_images):
        if i % 1000 == 0:
            print('{}/{}'.format(i, num_images))
        img_name = img_list[i]
        # read img
        img_path = os.path.join(img_dir, img_name)
        with open(img_path, 'rb') as fin:
            buf = fin.read()
        # write lst
        line = '%d\t0\t%s\n' % (i, img_name)
        lst_fn.write(line)
        # write rec
        s = mx.recordio.pack(mx.recordio.IRHeader(0, 0, i, 0), buf)
        imgrec.write_idx(i, s)
    lst_fn.close()
    imgrec.close()


def check_rec_from_img(img_dir, rec_prefix):
    img_list = read_list(rec_prefix + '.lst')
    imgrec = mx.recordio.MXIndexedRecordIO(rec_prefix + '.idx', rec_prefix + '.rec', 'r')
    num_images = len(img_list)
    for i, img_name in enumerate(img_list):
        if i % 100 == 0:
            print '%d/%d' % (i, num_images)
        _, img = mx.recordio.unpack_img(imgrec.read_idx(img_list[img_name]), cv2.IMREAD_COLOR)
        img1 = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_COLOR)
        assert np.sum(abs(img - img1)) == 0


def make_rec_from_video(video_path, save_rec_prefix):

    lst_fn = open(save_rec_prefix + '.lst', 'w')
    imgrec = mx.recordio.MXIndexedRecordIO(save_rec_prefix + '.idx', save_rec_prefix + '.rec', 'w')

    video_reader = imageio.get_reader(video_path)
    num_images = len(video_reader)

    for i in range(num_images):
        if i % 100 == 0:
            print('{}/{}'.format(i, num_images))
        img_name = '%08d.png' % i
        # read img
        img = np.array(video_reader.get_data(i))[:, :, ::-1]  # (RGB --> BGR)
        # write lst
        line = '%d\t0\t%s\n' % (i, img_name)
        lst_fn.write(line)
        # write rec
        ret, buf = cv2.imencode('.png', img)
        assert ret and np.sum(img - cv2.imdecode(buf, 1)) == 0
        s = mx.recordio.pack(mx.recordio.IRHeader(0, 0, i, 0), buf.tostring())
        imgrec.write_idx(i, s)
    lst_fn.close()
    imgrec.close()
    video_reader.close()


def check_rec_from_video(video_path, rec_prefix):
    img_list = read_list(rec_prefix + '.lst')
    imgrec = mx.recordio.MXIndexedRecordIO(rec_prefix + '.idx', rec_prefix + '.rec', 'r')
    num_images = len(img_list)
    video_reader = imageio.get_reader(video_path)
    for i, img_name in enumerate(img_list):
        if i % 100 == 0:
            print '%d/%d' % (i, num_images)
        # img_idx = img_list[img_name]
        img_idx = i
        _, img = mx.recordio.unpack_img(imgrec.read_idx(img_idx), cv2.IMREAD_COLOR)
        img1 = np.array(video_reader.get_data(img_idx))[:, :, ::-1]
        assert np.sum(abs(img - img1)) == 0
    imgrec.close()
    video_reader.close()


if __name__ == '__main__':
    img_dir = '/opt/hdfs/user/he.huang/project/helmet-det/dataset/helmet-data/pachong_data/pachong_images/'
    save_rec_prefix = 'pachong/train_val_test'
    make_rec_from_img(img_dir, save_rec_prefix)
    check_rec_from_img(img_dir, save_rec_prefix)
