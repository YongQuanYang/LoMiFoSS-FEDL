# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""

Authors: lvhaijun01@baidu.com
Date:     2019-06-30 00:10
"""
import logging
import re
#from auto_augment.autoaug.transform.data_augment_classifer_transform import PbtAutoAugmentClassiferTransform
#from auto_augment.autoaug.transform.autoaug_transform import AutoAugTransform
import numpy as np
import six
import cv2
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def scale_crop(
        input_size,
        scale_size=None,
        normalize=None,
        mode="crop"):
    """

    Args:
        input_size:
        scale_size:
        normalize:
        mode: "crop" or "no_crop"
    Returns:

    """
    if normalize is None:
        normalize = __imagenet_stats

    if mode == "crop":
        # transforms.Resize()接口， 输入szie单个值，则按长宽比将最小边缩放到size。
        # 若size是list, 则直接resize到对应尺寸。
        t_list = [
            #transforms.Resize((scale_size, scale_size)),
            transforms.Resize(scale_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ]
    elif mode == "no_crop":
        t_list = [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ]
    elif mode == "tta":
        t_list = [
            transforms.Resize(scale_size),
            transforms.RandomResizedCrop(input_size),
            # transforms.Resize((input_size, input_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ]

    print("t_list:{}".format(t_list))
    return transforms.Compose(t_list)


class PbaAugment(object):
    """
    pytorch 分类 PbaAugment transform
    """

    def __init__(
            self,
            input_size=224,
            scale_size=256,
            normalize=None,
            pre_transform=True,
            **kwargs):
        """

        Args:
            input_size:
            scale_size:
            normalize:
            pre_transform:
            **kwargs:
        """

        if normalize is None:
            self.normalize = __imagenet_stats
        else:
            self.normalize = normalize

        conf = kwargs["conf"]

        policy = kwargs["policy"]
        stage = "search"
        train_epochs = kwargs["hp_policy_epochs"]
        self.auto_aug_transform = AutoAugTransform.create(policy, stage=stage, train_epochs=train_epochs)
        #self.auto_aug_transform = PbtAutoAugmentClassiferTransform(conf)

        if pre_transform:
            self.pre_transform = transforms.Compose([
                transforms.Resize(scale_size),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.pre_transform = transforms.Resize(input_size)

        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
        self.cur_epoch = 0

    def set_epoch(self, indx):
        """

        Args:
            indx:

        Returns:

        """
        self.auto_aug_transform.set_epoch(indx)

    def reset_policy(self, new_hparams):
        """

        Args:
            new_hparams:

        Returns:

        """
        self.auto_aug_transform.reset_policy(new_hparams)

    def __call__(self, img):
        """

        Args:
            img: PIL image
        Returns:

        """
        # tensform resize
        if self.pre_transform:
            img = self.pre_transform(img)

        img = self.auto_aug_transform.apply(img)
        img = img.astype(np.uint8)
        img = self.post_transform(img)
        return img


def autoaugment_preproccess(
        input_size,
        scale_size,
        normalize=None,
        pre_transform=True,
        **kwargs):
    """

    Args:
        input_size:
        scale_size:
        normalize:
        pre_transform:
        **kwargs:

    Returns:

    """

    if normalize is None:
        normalize = __imagenet_stats

    augment = PbaAugment(
        input_size,
        scale_size,
        normalize=normalize,
        pre_transform=pre_transform,
        **kwargs)

    return augment


def inception_preproccess(input_size, normalize=None):
    """

    Args:
        input_size:
        normalize:

    Returns:

    """
    if normalize is None:
        normalize = __imagenet_stats
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ]
    )

def get_transform(
        augment=True,
        mode="normal",
        input_size=224,
        scale_size=256,
        **kwargs):
    """

    Args:
        augment:
        input_size:

    Returns:

    """
    normalize = __imagenet_stats
    # print("input_size:{}, scale_size:{}".format(input_size, scale_size))
    if augment:
        if mode == "auto_aug":
            return autoaugment_preproccess(
                input_size=input_size,
                scale_size=scale_size,
                normalize=normalize,
                **kwargs
            )
        elif mode == "no_crop":
            return scale_crop(
                input_size=input_size,
                scale_size=scale_size,
                normalize=normalize,
                mode=mode)
        elif mode == "crop":
            return scale_crop(
                input_size=input_size,
                scale_size=scale_size,
                normalize=normalize,
                mode=mode)
        elif mode == "inception":
            return inception_preproccess(
                input_size=input_size, normalize=normalize)
        else:
            assert 0

    else:
        return scale_crop(
            input_size=input_size,
            scale_size=scale_size,
            normalize=normalize,
            mode=mode)


class PicRecord(object):
    """
    PicRecord
    """

    def __init__(self, row):
        """

        Args:
            row:
        """
        self._data = row

    @property
    def sub_path(self):
        """

        Returns:

        """
        return self._data[0]

    @property
    def label(self):
        """

        Returns:

        """
        return self._data[1]


class PicReader(data.Dataset):
    """
    PicReader
    """

    def __init__(
            self,
            root_path,
            list_file,
            meta=False,
            transform=None,
            class_to_id_dict=None,
            cache_img=False,
            **kwargs):
        """

        Args:
            root_path:
            list_file:
            meta:
            transform:
            class_to_id_dict:
            cache_img:
            **kwargs:
        """

        self.root_path = root_path
        self.list_file = list_file
        self.transform = transform
        self.meta = meta
        self.class_to_id_dict = class_to_id_dict
        self.train_type = kwargs["conf"].get("train_type", "single_label")
        self.class_num = kwargs["conf"].get("class_num", 0)

        self._parse_list(**kwargs)
        self.cache_img = cache_img
        self.cache_img_buff = dict()
        if self.cache_img:
            self._get_all_img(**kwargs)

    def _get_all_img(self, **kwargs):
        """
        缓存图片进行预resize, 减少内存占用

        Returns:

        """

        scale_size = kwargs.get("scale_size", 256)

        for idx in range(len(self)):
            record = self.pic_list[idx]
            relative_path = record.sub_path
            if self.root_path is not None:
                image_path = os.path.join(self.root_path, relative_path)
            else:
                image_path = relative_path
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((scale_size, scale_size))
                self.cache_img_buff[image_path] = img
            except BaseException:
                print("img_path:{} can not by PILloaded".format(image_path))
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (scale_size, scale_size))
                    img = Image.fromarray(img)
                    self.cache_img_buff[image_path] = img
                except BaseException:
                    print("img_path:{} can not by cv2 loaded".format(image_path))

                pass

    def _load_image(self, directory):
        """

        Args:
            directory:

        Returns:

        """

        if not self.cache_img:
            if not os.path.isfile(directory):
                assert 0, "file not exist"
            img = Image.open(directory).convert('RGB')
        else:
            if directory in self.cache_img_buff:
                img = self.cache_img_buff[directory]
            else:
                img = Image.open(directory).convert('RGB')
        return img

    def _parse_list(self, **kwargs):
        """

        Args:
            **kwargs:

        Returns:

        """
        delimiter = kwargs.get("delimiter", " ")
        self.pic_list = []

        with open(self.list_file) as f:
            lines = f.read().splitlines()
            print(
                "PicReader:: found {} picture in `{}'".format(
                    len(lines), self.list_file))
            for i, line in enumerate(lines):
                record = re.split(delimiter, line)
                # record = line.split()
                assert len(record) == 2, "length of record is not 2!"

                if not os.path.splitext(record[0])[1]:
                    # 适配线上分类数据转无后缀的情况
                    record[0] = record[0] + ".jpg"

                #线上单标签情况兼容多标签，后续需去除
                record[1] = re.split(",", record[1])[0]

                self.pic_list.append(PicRecord(record))

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """
        record = self.pic_list[index]

        return self.get(record)

    def get(self, record):
        """

        Args:
            record:

        Returns:

        """
        relative_path = record.sub_path
        if self.root_path is not None:
            image_path = os.path.join(self.root_path, relative_path)
        else:
            image_path = relative_path

        img = self._load_image(image_path)

        process_data = self.transform(img)
        if self.train_type == "single_label":
            if self.class_to_id_dict:
                label = self.class_to_id_dict[record.label]
            else:
                label = int(record.label)
        elif self.train_type == "multi_labels":
            label_tensor = np.zeros((1, self.class_num))
            for label in record.label.split(","):
                label_tensor[0, int(label)] = 1
            label_tensor = np.squeeze(label_tensor)
            label = label_tensor

        if self.meta:
            return process_data, label, relative_path
        else:
            return process_data, label

    def __len__(self):
        """

        Returns:

        """
        return len(self.pic_list)

    def set_meta(self, meta):
        """

        Args:
            meta:

        Returns:

        """
        self.meta = meta

    def set_epoch(self, epoch):
        """

        Args:
            epoch:

        Returns:

        """
        self.transform.set_epoch(epoch)

    # only use in search
    def reset_policy(self, new_hparams):
        """

        Args:
            new_hparams:

        Returns:

        """
        if self.transform is not None:
            self.transform.reset_policy(new_hparams)


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        six.raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_file):
    """ Parse the classes file.
    """
    result = {}
    with open(csv_file) as csv_reader:
        for line, row in enumerate(csv_reader):
            try:
                class_id, class_name = row.strip().split(':')[:2]
                # print(class_id, class_name)
            except ValueError:
                six.raise_from(
                    ValueError(
                        'line {}: format should be \'class_id:class_name\''.format(line)),
                    None)

            class_id = _parse(
                class_id,
                int,
                'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError(
                    'line {}: duplicate class name: \'{}\''.format(
                        line, class_name))
            result[class_name] = class_id
    return result


def get_loaders(
        input_size,
        scale_size,
        workers=1,
        val_batch_size=16,
        train_batch_size=16,
        train_data_root="",
        val_data_root="",
        train_list="",
        val_list="",
        val_shuffle=False,
        train_shuffle=True,
        transform_mode="normal",
        val_mode="crop",
        autoaug_file=None,
        hp_policy_epochs=200,
        epoch_nums=0,
        is_train=True,
        use_class_map=False,
        cache_img=False,
        **kwargs
):
    """

    Args:
        val_batch_size:
        train_batch_size:
        input_size:
        workers:
        train_data_root:
        val_data_root:
        train_list:
        val_list:
        val_shuffle:
        train_shuffle:
        transform_mode="normal",
        autoaug_file=None

    Returns:

    """

    class_to_id_dict = {}

    if use_class_map:
        if os.path.isfile(os.path.join(train_data_root, "label_list.txt")):
            class_to_id_dict = _read_classes(
                os.path.join(train_data_root, "label_list.txt"))
        elif os.path.isfile(os.path.join(val_data_root, "label_list.txt")):
            class_to_id_dict = _read_classes(
                os.path.join(val_data_root, "label_list.txt"))
        else:
            assert 0
    if not val_list:
        assert 0, "val_list:{} is not exist".format(val_list)
    else:
        val_data = PicReader(
            root_path=val_data_root,
            list_file=val_list,
            transform=get_transform(
                augment=False,
                input_size=input_size,
                scale_size=scale_size,
                mode=val_mode),
            class_to_id_dict=class_to_id_dict,
            cache_img=cache_img,
            scale_size=scale_size,
            **kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=workers,
        pin_memory=True)

    if is_train:
        if train_list is None:
            assert 0, "train_list:{} is not exist".format(train_list)
        else:
            train_data = PicReader(
                root_path=train_data_root,
                list_file=train_list,
                transform=get_transform(
                    mode=transform_mode,
                    auto_file=autoaug_file,
                    hp_policy_epochs=hp_policy_epochs,
                    epoch_nums=epoch_nums,
                    input_size=input_size,
                    scale_size=scale_size,
                    **kwargs),
                class_to_id_dict=class_to_id_dict,
                cache_img=cache_img,
                scale_size=scale_size,
                **kwargs)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            num_workers=workers,
            pin_memory=True)
    else:
        train_loader = None
    return train_loader, val_loader
