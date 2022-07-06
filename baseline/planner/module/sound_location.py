import json
import os
import tempfile

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision


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


class SoundPosition(nn.Module):
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

class SoundLocationPredictor:
    def __init__(self, learned='all'):
        self.model = SoundPosition(n_output=2 if learned == 'all' else 3).cuda()
        self.learned = learned
        model = {
            'all': 'pretrained/location_all.pth.tar',
            'craftroom': 'planner/sound_localization_code/craftroom_Epoch56_checkpoint.pth.tar',
            'kitchen': 'planner/sound_localization_code/kitchen_Epoch43_checkpoint.pth.tar'
        }[learned]
        self.model.load_state_dict(torch.load(model)['state_dict'])
        self.model.eval()

    def predict_location(self, audio):
        spec = process_audio(audio, self.learned)[None].cuda()
        position = self.model(spec).detach().cpu().numpy()[0]
        x, z = {
            'all': (15.67631186167073, 6.91433059411295),
            'kitchen': (5.751120625071255, 6.0083216805870014),
            'craftroom': (6.2716995103908175, 5.634416866119664),
        }[self.learned]
        return position[0] * x, position[1] * z


if __name__ == '__main__':
    with open('planner/sound_localization_code/distractor_data_list/train_list_distractor.txt') as f:
        cases = f.readlines()
    for c in sorted(cases):
        c = c.strip()
        print(c)
        position = predict_location(f'perception_dis/{c}.wav')
        with open(f'perception_dis/{c}.json') as f:
            loc = json.loads(f.read())['position']
            print(position)
            print(loc[0], loc[-1])
            print(((loc[0] / 15.67631186167073 - position[0] / 15.67631186167073)**2+(loc[1] / 6.91433059411295 - position[1] / 6.91433059411295)**2)**.5)
