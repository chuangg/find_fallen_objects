import os
import tempfile

import cv2
import librosa
import scipy.special

import torch.nn as nn
import torchvision
import torch
import json
import numpy as np

def process_audio(audio, learned='all'):
    delete = False
    if isinstance(audio, np.ndarray):
        temp = tempfile.mkstemp('.wav')[1]
        with open(temp, 'wb') as f:
            f.write(audio.tobytes().rstrip(b'\x01'))
        audio = temp
        delete = True
    window_size = [25, 50, 100]
    hop_size = [10, 15, 20]
    no_channel = [0, 1, 0]
    y, sr = librosa.load(audio, mono=False)
    if delete:
        os.remove(audio)
    y = (y - y.mean()) / y.std()
    raw_spec = [librosa.feature.melspectrogram(y=y[c], sr=sr, n_fft=sr//10, win_length=w*sr//2000, hop_length=h*sr//2000, n_mels=224) for w, h, c in zip(window_size, hop_size, no_channel)]
    specs = []
    for spec in raw_spec:
        spec = (spec - spec.mean()) / spec.std()
        spec = cv2.resize(spec, (224, 224))
        specs.append(spec)
    specs = np.array(specs, dtype=np.float32)
    mean, std = {
        'all': (-1.59605e-05, 0.94158906),
        'craftroom': (-2.7876264e-05, 0.9420827),
        'kitchen': (8.389836e-05, 0.9487269),
    }[learned]
    specs = (specs - mean) / std
    return torch.from_numpy(specs)

CATEGORY_IDX = {
  "bookend": 0,
  "bottle": 1,
  "bottle cork": 2,
  "bowl": 3,
  "box": 4,
  "calculator": 5,
  "candle": 6,
  "clothesbrush": 7,
  "coaster": 8,
  "cup": 9,
  "flashlight battery": 10,
  "fork": 11,
  "golf ball": 12,
  "headphone": 13,
  "ipod": 14,
  "jug": 15,
  "key": 16,
  "kitchen utensil": 17,
  "knife": 18,
  "pen": 19,
  "pepper mill, pepper grinder": 20,
  "shirt button": 21,
  "soda can": 22,
  "spoon": 23,
  "toaster": 24,
  "toothbrush": 25,
  "toy": 26,
  "vase": 27,
  "watch": 28,
  "wineglass": 29
}

CATE_MAP = {'cabinet': (67, 35, 17), 'padlock': (227, 74, 201), 'dog house': (11, 130, 245), 'skate': (164, 155, 28), 'sofa': (218, 201, 130), 'carving fork': (14, 102, 232), 'refrigerator': (200, 95, 119), 'toy': (86, 119, 40), 'garden rake': (156, 41, 66), 'cup': (49, 197, 183), 'coffee grinder': (137, 242, 231), 'beverage': (176, 80, 56), 'spoon': (139, 80, 65), 'knife': (176, 71, 29), 'pan': (70, 230, 179), 'comb': (242, 139, 41), 'television set': (112, 159, 97), 'teakettle': (100, 15, 119), 'watch': (15, 218, 148), 'camera': (174, 137, 80), 'alarm clock': (14, 3, 24), 'globe': (193, 65, 250), 'sculpture': (51, 8, 136), 'lighter': (120, 87, 207), 'trunk': (164, 226, 24), 'pencil': (98, 232, 2), 'fork': (54, 101, 3), 'chair': (198, 90, 222), 'table': (205, 108, 238), 'bottle': (45, 76, 225), 'computer mouse': (111, 168, 50), 'rug': (94, 249, 79), 'bag, handbag, pocketbook, purse': (29, 234, 145), 'pliers': (145, 7, 241), 'hammer': (252, 89, 22), 'cassette': (68, 95, 1), 'bed': (164, 204, 252), 'painting': (211, 40, 182), 'clothesbrush': (85, 49, 232), 'vase': (33, 183, 49), 'table lamp': (81, 200, 174), 'jug': (69, 99, 14), 'pot': (169, 163, 122), 'shelf': (13, 183, 123), 'headphone': (26, 80, 230), 'bee': (152, 151, 174), 'bookend': (26, 220, 101), 'coffee maker': (172, 134, 103), 'basket': (76, 142, 162), 'toaster': (54, 142, 213), 'pepper mill, pepper grinder': (39, 200, 155), 'scissors': (172, 243, 59), 'trophy': (209, 97, 188), 'lawn mower, mower': (140, 212, 91), 'ipod': (212, 194, 173), 'hairbrush': (33, 104, 25), 'bottle cork': (211, 26, 46), 'soda can': (255, 113, 220), 'kitchen utensil': (18, 14, 222), 'throw pillow': (195, 9, 244), 'printer': (20, 103, 83), 'backpack': (249, 251, 219), 'book': (201, 225, 158), 'vacuum cleaner': (182, 47, 21), 'laptop, laptop computer': (5, 218, 110), 'wineglass': (238, 232, 36), 'cookie sheet': (29, 170, 74), 'candle': (170, 31, 201), 'remote': (65, 163, 219), 'bowl': (174, 174, 241), 'pen': (168, 60, 40), 'floor lamp': (70, 226, 24), 'coffee table, cocktail table': (224, 246, 73), 'dishwasher': (47, 23, 44), 'shirt button': (73, 103, 47), 'cog, gear': (44, 131, 34), 'key': (255, 44, 0), 'screwdriver': (160, 74, 39), 'coaster': (119, 157, 128), 'golf ball': (70, 77, 193), 'box': (210, 159, 67), 'suitcase': (2, 112, 31), 'picture': (71, 104, 221), 'calculator': (164, 198, 133), 'flashlight battery': (121, 17, 248), 'microwave, microwave oven': (99, 15, 83), 'toothbrush': (90, 10, 212), 'saltshaker, salt shaker': (186, 115, 57)}

CATE_IDX_COLOR = np.zeros((30, 3), dtype=np.uint8)
for (k, idx) in CATEGORY_IDX.items():
    CATE_IDX_COLOR[idx] = CATE_MAP[k]

class SoundCategory(nn.Module):
    def __init__(self, n_output=4, finetune=True, num_fc=3, resnet_type='resnet50' ):
        super().__init__()
        if resnet_type=='resnet50':
            self.resnet = torchvision.models.resnet50(pretrained=True)
        elif resnet_type=='resnet18':
            self.resnet = torchvision.models.resnet18(pretrained=True)
        if not finetune:
            for p in self.resnet.parameters():
                p.requires_grad = False
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(self.resnet.fc.in_features, 1024), nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(1024, n_output),
        # )
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential()

        self.resnet.fc.add_module('fc-1',nn.Linear(in_features, 1024))
        self.resnet.fc.add_module('fc-1-Relu',nn.ReLU())
        self.resnet.fc.add_module('fc-1-Dropout',nn.Dropout(0.4))

        for i in range(num_fc-2):
            self.resnet.fc.add_module('fc-{}'.format(i+2),nn.Linear(1024, 1024))
            self.resnet.fc.add_module('fc-{}-Relu'.format(i+2),nn.ReLU())
            self.resnet.fc.add_module('fc-{}-Dropout'.format(i+2),nn.Dropout(0.4))

        self.resnet.fc.add_module('fc-{}'.format(num_fc),nn.Linear(1024, n_output))
        print(self.resnet.fc)
        # if not finetune:
        #     for p in self.resnet.fc.parameters():
        #         p.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


class SoundCategoryPredictor:
    def __init__(self, learned='all'):
        self.model = SoundCategory(n_output=30).cuda()
        self.learned = learned
        if learned == 'all':
            self.model.load_state_dict(torch.load('pretrained/category_all.pth.tar')['state_dict'])
        elif learned == 'craftroom':
            self.model.load_state_dict(torch.load('ss_baselines/Epoch28_checkpoint_craftroom.pth.tar')['state_dict'])
        elif learned == 'kitchen':
            self.model.load_state_dict(torch.load('ss_baselines/Epoch122_checkpoint_kitchen.pth.tar')['state_dict'])
        else:
            assert False
        self.model.eval()

    def predict_category(self, audio):
        spec = process_audio(audio, self.learned)[None].cuda()
        category = self.model(spec).detach().cpu().numpy()[0]
        return scipy.special.softmax(category, axis=-1)


if __name__ == '__main__':
    with open('data_list/test.txt') as f:
        cases = f.readlines()
    for c in sorted(cases):
        c = c.strip()
        print(c)
        category = predict_category(f'perception/{c}.wav')
        with open(f'perception/{c}.json') as f:
            loc = json.loads(f.read())['category']
            print(f'{category[CATEGORY_IDX[loc]]:.3f}', len(category[category > 0.05]))
        # with open(f'perception_dis/{c}.json') as f:
        #     loc = json.loads(f.read())['position']
        #     print(position)
        #     print(loc[0], loc[-1])
        #     print(((loc[0] / 15.67631186167073 - position[0] / 15.67631186167073)**2+(loc[1] / 6.91433059411295 - position[1] / 6.91433059411295)**2)**.5)
