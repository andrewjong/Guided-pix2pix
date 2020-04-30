import torch.nn.functional as F
import json
import os.path
from glob import glob
import os.path as osp
import torchvision.transforms as transforms
import torch
from PIL import ImageDraw
from PIL import Image

from data.base_dataset import BaseDataset
import numpy as np
import pickle
import random

# value 1 and -1
# for circles

# 9
# pixels
# wide;
# radius is 4


class VvtCompetitionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.radius = 4
        self.opt = opt
        self.root = opt.dataroot
        self._clothes_person_dir = osp.join(self.root, "lip_clothes_person")

        self.img_h = 256
        self.img_w = 192

        if opt.isTrain:
            self._keypoints_dir = osp.join(self.root, "lip_train_frames_keypoint")
            raise NotImplementedError(
                "Train is not yet implemented for vvt competition")
        else:
            self._keypoints_dir = osp.join(self.root, "lip_test_frames_keypoint")

        self.keypoints = glob(f"{self._keypoints_dir}/**/*.json")
        assert len(self.keypoints > 0)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def randomFlip(self, x, x_target, pose, pose_target, mask, mask_target):
        # random horizontal flip
        rand = random.randint(0, 1)
        if (rand == 1):
            x = np.flip(x, axis=1).copy()
            x_target = np.flip(x_target, axis=1).copy()
            pose = np.flip(pose, axis=1).copy()
            pose_target = np.flip(pose_target, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            mask_target = np.flip(mask_target, axis=1).copy()
        return x, x_target, pose, pose_target, mask, mask_target

    def get_input_person_pose(self, index, target_width):
        """from cp-vton, loads the pose as white squares
        returns pose map, image of pose map
        """
        # load pose points
        _pose_name = self.keypoints[index]
        with open(osp.join(self.data_path, 'pose', _pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]  # how many pose joints
        assert len(point_num) == 18, "should be 18 pose joints for guidedpix2pix"
        # construct an N-channel map tensor
        pose_map = torch.zeros(point_num, self.img_h, self.img_w)
        pose_map -= 1  # set -1 everywhere

        r = self.radius

        # draw a circle around the joint on the appropriate channel
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            one_map_tensor = self.to_tensor_and_norm(one_map)
            pose_map[i] = one_map_tensor[0]

            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.ellipse((pointx - r, pointy - r, pointx + r, pointy + r), 1, 1)

        # add padding to the w/ h/
        pad = (target_width - self.img_w)//2
        F.pad(pose_map, (pad, pad))  # make the image 256x256
        return pose_map

    def get_person_image(self, index):
        pose_name = self.keypoints[index]
        person_id = osp.split(pose_name)[-2]
        folder = osp.join(self._clothes_person_dir, person_id)

        person_image_path = os.listdir(folder)[-1] # TODO: verify my index
        assert person_image_path.endswith(".png"), f"person images should have .png extensions: {person_image_path}"
        person_image = Image.open(person_image_path)
        return person_image


    def __getitem__(self, index):
        image = self.get_person_image(index)  # (256, 256, 3)
        pose_target = self.get_input_person_pose(index, target_width=256)  # (256, 256, 18)

        # random fliping
        # if (self.opt.isTrain):
        #     x, x_target, pose, pose_target, mask, mask_target = self.randomFlip(x,
        #                                                                         x_target,
        #                                                                         pose,
        #                                                                         pose_target,
        #                                                                         mask,
        #                                                                         mask_target)

        # to tensor
        image = self.transform(image)  # ranges from [0, 255]
        pose_target = self.transform(pose_target)

        # input-guide-target
        input = (image / 255) * 2 - 1
        guide = pose_target
        # target = (x_target / 255) * 2 - 1
        # Put data into [input, guide, target]
        return {'A': input, 'guide': guide, 'B': None}

    def __len__(self):
        return len(self.keypoints)

    def name(self):
        return 'VvtCompetitionDataset'
