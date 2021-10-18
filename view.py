import glob
from plyfile import PlyData, PlyElement
import numpy as np
import torch


files = sorted(glob.glob('/home/aidrive1/workspace/luoly/dataset/Min_scan/scan_processed/test/*.ply'))
files2 = sorted(glob.glob('/home/aidrive1/workspace/luoly/dataset/final/test/20/*.txt'))

VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

assert len(files) == len(files2)
for i in range(len(files)):
    a = PlyData.read(files[i])
    b = np.loadtxt(files2[i], dtype=int)
    print(len(b))
    for k in range(len(b)):
        colors = np.array(COLOR_MAP[b[k]])
        a['vertex']['red'][k] = colors[0]
        a['vertex']['green'][k] = colors[1]
        a['vertex']['blue'][k] = colors[2]
    a.write('/home/aidrive1/workspace/luoly/dataset/final/view/20/' +
            files[i][-16:-4]+'_test.ply')
