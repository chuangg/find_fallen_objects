import os

import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

category_color = {
    'cabinet': (67, 35, 17), 'padlock': (227, 74, 201), 'dog house': (11, 130, 245),
    'skate': (164, 155, 28), 'sofa': (218, 201, 130), 'carving fork': (14, 102, 232),
    'refrigerator': (200, 95, 119), 'toy': (86, 119, 40), 'garden rake': (156, 41, 66),
    'cup': (49, 197, 183), 'coffee grinder': (137, 242, 231), 'beverage': (176, 80, 56),
    'spoon': (139, 80, 65), 'knife': (176, 71, 29), 'pan': (70, 230, 179),
    'comb': (242, 139, 41), 'television set': (112, 159, 97), 'teakettle': (100, 15, 119),
    'watch': (15, 218, 148), 'camera': (174, 137, 80), 'alarm clock': (14, 3, 24),
    'globe': (193, 65, 250), 'sculpture': (51, 8, 136), 'lighter': (120, 87, 207),
    'trunk': (164, 226, 24), 'pencil': (98, 232, 2), 'fork': (54, 101, 3),
    'chair': (198, 90, 222), 'table': (205, 108, 238), 'bottle': (45, 76, 225),
    'computer mouse': (111, 168, 50), 'rug': (94, 249, 79), 'bag, handbag, pocketbook, purse': (29, 234, 145),
    'pliers': (145, 7, 241), 'hammer': (252, 89, 22), 'cassette': (68, 95, 1),
    'bed': (164, 204, 252), 'painting': (211, 40, 182), 'clothesbrush': (85, 49, 232),
    'vase': (33, 183, 49), 'table lamp': (81, 200, 174), 'jug': (69, 99, 14),
    'pot': (169, 163, 122), 'shelf': (13, 183, 123), 'headphone': (26, 80, 230),
    'bee': (152, 151, 174), 'bookend': (26, 220, 101), 'coffee maker': (172, 134, 103),
    'basket': (76, 142, 162), 'toaster': (54, 142, 213), 'pepper mill, pepper grinder': (39, 200, 155),
    'scissors': (172, 243, 59), 'trophy': (209, 97, 188), 'lawn mower, mower': (140, 212, 91),
    'ipod': (212, 194, 173), 'hairbrush': (33, 104, 25), 'bottle cork': (211, 26, 46),
    'soda can': (255, 113, 220), 'kitchen utensil': (18, 14, 222), 'throw pillow': (195, 9, 244),
    'printer': (20, 103, 83), 'backpack': (249, 251, 219), 'book': (201, 225, 158),
    'vacuum cleaner': (182, 47, 21), 'laptop, laptop computer': (5, 218, 110), 'wineglass': (238, 232, 36),
    'cookie sheet': (29, 170, 74), 'candle': (170, 31, 201), 'remote': (65, 163, 219),
    'bowl': (174, 174, 241), 'pen': (168, 60, 40), 'floor lamp': (70, 226, 24),
    'coffee table, cocktail table': (224, 246, 73), 'dishwasher': (47, 23, 44), 'shirt button': (73, 103, 47),
    'cog, gear': (44, 131, 34), 'key': (255, 44, 0), 'screwdriver': (160, 74, 39), 'coaster': (119, 157, 128),
    'golf ball': (70, 77, 193), 'box': (210, 159, 67), 'suitcase': (2, 112, 31),
    'picture': (71, 104, 221), 'calculator': (164, 198, 133), 'flashlight battery': (121, 17, 248),
    'microwave, microwave oven': (99, 15, 83), 'toothbrush': (90, 10, 212), 'saltshaker, salt shaker': (186, 115, 57)
}

key_category_dict = {
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

CATEGORIES = {0: (0, 0, 0)}
for (k, i) in key_category_dict.items():
    CATEGORIES[i] = category_color[k]

class SegMaskPredictor:
    def __init__(self, device, learned='all'):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        cfg.OUTPUT_DIR = 'pretrained'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 30
        if learned == 'all':
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'segm_all.pth')
        elif learned == 'craftroom':
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_craftroom_0189999.pth')
        elif learned == 'kitchen':
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_kitchen_0189999.pth')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.DEVICE = f'cuda:{device}'
        self.predictor = DefaultPredictor(cfg)

    def get_cate_mask(self, image):
        outputs = self.predictor(image[:, :, ::-1]) # the model uses bgr for input
        pred_masks = outputs['instances'].to('cpu').get_fields()['pred_masks']
        pred_classes = outputs['instances'].to('cpu').get_fields()['pred_classes']
        pred_pixel = [CATEGORIES[x] for x in pred_classes.tolist()]

        mask = np.zeros_like(image).reshape(-1, 3)
        for ind, cls in enumerate(pred_classes):
            mask_f = pred_masks[ind].reshape(-1, 1).squeeze()
            mask[mask_f, :] = pred_pixel[ind]
        mask = mask.reshape(image.shape)

        return mask
