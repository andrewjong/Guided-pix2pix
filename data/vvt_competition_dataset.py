import torch.nn.functional as F
from skimage import draw
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
        self.radius = 4.5
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
        assert len(self.keypoints) > 0

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

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
        with open(_pose_name, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]  # how many pose joints
        assert point_num == 18, "should be 18 pose joints for guidedpix2pix"
        # construct an N-channel map tensor with -1
        pose_map = torch.zeros(point_num, self.img_h, self.img_w) - 1

        # draw a circle around the joint on the appropriate channel
        for i in range(point_num):
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                rr, cc = draw.circle(pointy, pointx, self.radius, shape=(self.img_h, self.img_w))
                pose_map[i, rr, cc] = 1
                # assert torch.any(pose_map[i] == 1), "drew but did not find 1s"

        # add padding to the w/ h/
        pad = (target_width - self.img_w)//2
        pose_map = F.pad(pose_map, (pad, pad), value=-1)  # make the image 256x256
        assert all(i == -1 or i == 1 for i in torch.unique(pose_map)), f"{torch.unique(pose_map)}"
        return pose_map

    def _get_person_image_path_from_index(self, index):
        pose_name = self.keypoints[index]
        person_id = pose_name.split("/")[-2]
        folder = osp.join(self._clothes_person_dir, person_id)

        files = os.listdir(folder)
        person_image_name = [f for f in files if f.endswith(".png")][0]
        assert person_image_name.endswith(".png"), f"person images should have .png extensions: {person_image_name}"
        return osp.join(folder, person_image_name)

    def get_person_image(self, index):
        pers_image_path = self._get_person_image_path_from_index(index)
        person_image = Image.open(pers_image_path)
        person_tensor = self.to_tensor(person_image)
        pad = (256 - 192) // 2
        person_tensor = F.pad(person_tensor, (pad, pad))
        return person_tensor


    def __getitem__(self, index):
        image = self.get_person_image(index)  # (3, 256, 256)
        pose_target = self.get_input_person_pose(index, target_width=256)  # (18, 256, 256)
        assert image.shape[-2:] == pose_target.shape[-2:], f"hxw don't match: image {image.shape}, pose {pose_target.shape}"

        # random fliping
        # if (self.opt.isTrain):
        #     x, x_target, pose, pose_target, mask, mask_target = self.randomFlip(x,
        #                                                                         x_target,
        #                                                                         pose,
        #                                                                         pose_target,
        #                                                                         mask,
        #                                                                         mask_target)

        # input-guide-target
        input = (image / 255) * 2 - 1
        guide = pose_target
        # target = (x_target / 255) * 2 - 1
        # Put data into [input, guide, target]
        return {'A': input, "guide_path": self.keypoints[index], 'guide': guide, 'B': torch.zeros(3, 256, 256)}

    def __len__(self):
        return len(self.keypoints)

    def name(self):
        return 'VvtCompetitionDataset'
