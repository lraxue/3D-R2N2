# -*- coding: utf-8 -*-
# @Time    : 17-10-27 下午4:32
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : fei_test.py
# @Software: PyCharm Community Edition

from lib import binvox_rw as binvoxel
import numpy as np


def test_voxel(voxels=None):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    # print(cube_verts)
    # print(cube_faces)


def load_voxel(file):
    with open(file, 'rb') as f:
        voexl_array = binvoxel.read_as_3d_array(f)

    print(voexl_array.dims)
    print(voexl_array.scale)

    voexl_array = np.array(voexl_array.data.flatten())

    # for i in range(32):
    #     for j in range(32):
    #         for k in range(32):
    #             print(i, j, k, voexl_array[i * 32 * 32 + j * 32 + k])


def main():
    voxel_name = 'model.binvox'
    load_voxel(voxel_name)



if __name__ == '__main__':
    main()