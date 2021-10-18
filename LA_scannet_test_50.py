# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import numpy as np
from urllib.request import urlretrieve
try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')
from pathlib import Path
import torch
import MinkowskiEngine as ME
from sklearn.neighbors import NearestNeighbors
from collections import  Counter
from models.res16unet import Res16UNet34C

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

parser = argparse.ArgumentParser()
parser.add_argument('--weights1', type=str,
                    default='/home/aidrive1/workspace/luoly/dataset/final/min_weights/unc/unc_50/checkpoint_NoneRes16UNet34Cbest_val.pth')
parser.add_argument('--weights2', type=str,
                    default='/home/aidrive1/workspace/luoly/dataset/final/min_weights/bt_eff/512/50/checkpoint_NoneRes16UNet34Cbest_val.pth')
parser.add_argument('--weights3', type=str,
                    default='/home/aidrive1/workspace/luoly/dataset/final/min_weights/bt_eff/1024/50/checkpoint_NoneRes16UNet34Cbest_val.pth')
parser.add_argument('--file_name', type=str, default='1.ply')
parser.add_argument('--bn_momentum', type=float, default=0.05)
parser.add_argument('--voxel_size', type=float, default=0.02)
parser.add_argument('--conv1_kernel_size', type=int, default=5)

SCANNET_DATA_RAW_PATH = Path(
    '/home/aidrive1/workspace/luoly/dataset/scn/scan_dataset/')
SCANNET_OUT_PATH = Path(
    '/home/aidrive1/workspace/luoly/dataset/final/test_Data/')
TEST_DEST = 'test'
SUBSETS = {TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'


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


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


data_paths = read_txt('./splits/scannet/scannetv2_test.txt')
data_paths = sorted(data_paths)
path_list = []
for out_path, in_path in SUBSETS.items():
    for f in (SCANNET_DATA_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
        a = str(f.parent.name)
        if a in data_paths:
            path_list.append(str(f))
path_list = sorted(path_list)


def load_file(file_name, voxel_size):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    feats = np.array(pcd.colors)

    quantized_coords = np.floor(coords / voxel_size)
    inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)

    return quantized_coords[inds[1]], feats[inds[1]], inds[1]


def generate_input_sparse_tensor(file_name, voxel_size=0.05):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(file_name, voxel_size)]
    coordinates_, featrues_, inds = list(zip(*batch))
    #coordinates_, featrues_, pcds = load_file(file_name, voxel_size)
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

    # Normalize features and create a sparse tensor
    return coordinates, (features - 0.5).float(), inds


config = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a model and load the weights
model1 = Res16UNet34C(3, 20, config).to(device)
model2 = Res16UNet34C(3, 20, config).to(device)
model3 = Res16UNet34C(3, 20, config).to(device)
model_dict1 = torch.load(config.weights1)
model_dict2 = torch.load(config.weights1)
model_dict3 = torch.load(config.weights1)
model1.load_state_dict(model_dict1['state_dict'])
model2.load_state_dict(model_dict2['state_dict'])
model3.load_state_dict(model_dict3['state_dict'])
model1.eval()
model2.eval()
model3.eval()

for i in range(0, len(path_list)):
    # Measure time
    with torch.no_grad():
        coordinates, features, inds = generate_input_sparse_tensor(
            path_list[i], voxel_size=config.voxel_size)

        # Feed-forward pass and get the prediction
        sinput = ME.SparseTensor(features.to(
            device), coordinates=coordinates.int().to(device))
        soutput1 = model1(sinput)
        soutput2 = model2(sinput)
        soutput3 = model3(sinput)
    print(str(path_list[i])[-27:-15])
    # Feed-forward pass and get the prediction
    output = (soutput1.F + soutput2.F + soutput3.F) / 3
    #_, pred = soutput.F.max(1)
    _, pred = output.max(1)
    pred = pred.cpu().numpy()
    inds = inds[0].cpu().numpy()
    #np.save('/home/aidrive1/workspace/luoly/dataset/final/eff_20/unc/pseudo_label_%s.npy' %
    #        (str(path_list[i])[-27:-15]), pred)
    #np.save('/home/aidrive1/workspace/luoly/dataset/final/eff_20/unc/map_%s.npy' %
    #        (str(path_list[i])[-27:-15]), inds)
    path = '/home/aidrive1/workspace/luoly/dataset/final/LA/50/' + str(path_list[i])[-27:-15] + '.txt'
    coordinate = coordinates[:, 1:].int()
    pcd = o3d.io.read_point_cloud(path_list[i])
    coords = np.array(pcd.points)
    coord = np.floor(coords / config.voxel_size)
    neigh = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(coordinate)
    dist, indexes = neigh.kneighbors(coord, return_distance=True)

    a = np.ones(len(indexes))*(40)
    print(len(indexes))
    for i, x in enumerate(inds):
      a[x] = pred[i]
    print(sum(a!=40))
    a = a.astype(int)
    for m in range(len(a)):
      if m in inds:
        a[m] = a[m]
      else:
        a[m] = pred[indexes[m][0]]
  
    for m in range(len(a)):   
        a[m] = VALID_CLASS_IDS[a[m]]
    print(sum(a!=40))
    #print(sum(a==39))
    #print(sum(a==36))
    #print(sum(a==34))
    #print(sum(a==33))
    #print(sum(a==28))
    #print(sum(a==24))
    #print(sum(a==16))
    #print(sum(a==14))
    #print(sum(a==12))
    #print(sum(a==11))
    #print(sum(a==10))
    #print(sum(a==9))
    #print(sum(a==8))
    #print(sum(a==7))
    #print(sum(a==6))
    #print(sum(a==5))
    #print(sum(a==4))
    #print(sum(a==3))
    #print(sum(a==2))
    #print(sum(a==1))

    fp = open(path , 'w')
    for j in range(len(a)):
      if j == len(a) - 1:
        fp.write('{}'.format(a[j]))
      else:
        fp.write('{}\n'.format(a[j]))
    fp.close()
    #np.savetxt(path, a, fmt="%d", delimiter="\n")




