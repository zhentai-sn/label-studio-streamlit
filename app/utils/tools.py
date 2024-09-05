
import streamlit as st
from PIL import Image
import json
import io
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import glob
import os
from datetime import datetime
import json
import uuid

from app.frontend import st_label_studio


def generate_unique_id():
    return str(uuid.uuid4())

def get_config():
    label_studio_app_config = json.loads(io.open('app/label_studio_configs/weiui_config.json', 'r', encoding='utf8').read())

    description = label_studio_app_config['description']
    interfaces = label_studio_app_config['interfaces']
    user = label_studio_app_config['user']
    task = label_studio_app_config['task']
    config_path = label_studio_app_config['config']

    

    # 读取XML文件
    tree = ET.parse(config_path)
    config_x = tree.getroot()
    config_x = ET.tostring(config_x, encoding='unicode')
    

    return description, config_x, interfaces, user, task

def edd_cvt_lsf(temp_file_path, edd_position, task,gbm_mask):
    img = cv2.imread(temp_file_path)
    img_h, img_w, _ = img.shape
    edd_threshold = 0.9
    edd_scores = edd_position.scores[edd_position.scores >= edd_threshold]
    edd_bboxes = edd_position.bboxes[edd_position.scores >= edd_threshold]
    edd_labels = edd_position.labels[edd_position.scores >= edd_threshold]

    # 坐标归一化
    edd_bboxes[:, [0, 2]] /= img_w
    edd_bboxes[:, [1, 3]] /= img_h
    edd_bboxes *=100
    edd_bboxes = edd_bboxes.cpu().numpy().astype(np.float64)

    # 坐标格式转化
    x_list,  y_list= edd_bboxes[:,0], edd_bboxes[:,1]
    height_list  =  edd_bboxes[:, 2] - edd_bboxes[:, 0]  # Calculate width
    width_list =  edd_bboxes[:, 3] - edd_bboxes[:, 1]  # Calculate height

    # 更新task
    for x,y,h,w in zip(x_list, y_list, height_list, width_list):

        time_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")+ str(datetime.now().microsecond)
        task['annotations'][0]['result'].append({   
                            "original_width": 100,
                            "original_height": 100,
                            "from_name": "tag",
                            "id": f"{time_id}",
                            "source": "$image",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "rectanglelabels": ["EDD"],
                                "rotation": 0,

                            }
                            
                        })
    rle = mask2rle(gbm_mask) 
    task['annotations'][0]['result'].append(
        {
            "id":"123",
            "from_name":"labels",
            "to_name":"image",
            "type":"brushlabels",
            "origin":"manual",
            "value":{
                "format": "rle",
                "rle":rle,
                "brushlabels": ["GBM"]
            }
        }
    )
  
    return task
    
def bits2byte(arr_str, n=8):
    """Convert bits back to byte

    :param arr_str:  string with the bit array
    :type arr_str: str
    :param n: number of bits to separate the arr string into
    :type n: int
    :return rle:
    :type rle: list
    """
    rle = []
    numbers = [arr_str[i : i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle

def base_rle_encode(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]

def encode_rle(arr, wordsize=8, rle_sizes=[3, 4, 8, 16]):


    # 用32位设置数组的长度
    num = len(arr)
    numbits = f'{num:032b}'

    # 设置wordsize的位数
    wordsizebits = f'{wordsize - 1:05b}'

    # 将rle_sizes转换为位数
    rle_bits = ''.join([f'{x - 1:04b}' for x in rle_sizes])

    # 将这些部分组合成基础字符串
    base_str = numbits + wordsizebits + rle_bits

    # 开始创建RLE位字符串
    out_str = ''
    for length_reeks, p, value in zip(*base_rle_encode(arr)):
        # TODO: 这部分可以优化，但目前功能正常
        if length_reeks == 1:
            # 表示该数值的长度为1，用第一个0表示
            out_str += '0'
            # 用00表示rle_sizes中的索引
            out_str += '00'
            # rle_size值为0，表示单个数字
            out_str += '000'
            # 将数值转换为8位字符串
            out_str += f'{value:08b}'
            state = 'single_val'

        elif length_reeks > 1:
            state = 'series'
            # rle size = 3
            if length_reeks <= 8:
                # 用1表示开始一个系列
                out_str += '1'
                # rle_sizes数组中的索引
                out_str += '00'
                # 将系列长度转换为位
                out_str += f'{length_reeks - 1:03b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

            # rle size = 4
            elif 8 < length_reeks <= 16:
                # 用1表示开始一个系列
                out_str += '1'
                out_str += '01'
                # 将系列长度转换为位
                out_str += f'{length_reeks - 1:04b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

            # rle size = 8
            elif 16 < length_reeks <= 256:
                # 用1表示开始一个系列
                out_str += '1'
                out_str += '10'
                # 将系列长度转换为位
                out_str += f'{length_reeks - 1:08b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

            # rle size = 16或更长
            else:
                length_temp = length_reeks
                while length_temp > 2**16:
                    # 用1表示开始一个系列
                    out_str += '1'
                    out_str += '11'
                    out_str += f'{2**16 - 1:016b}'
                    out_str += f'{value:08b}'
                    length_temp -= 2**16

                # 用1表示开始一个系列
                out_str += '1'
                out_str += '11'
                # 将系列长度转换为位
                out_str += f'{length_temp - 1:016b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

    # 确保最终字符串长度为8的倍数，不足时在末尾补0
    nzfill = 8 - len(base_str + out_str) % 8
    total_str = base_str + out_str
    total_str = total_str + nzfill * '0'

    # 将位字符串转换为字节
    rle = bits2byte(total_str)

    return rle

def mask2rle(mask):
    """Convert mask to RLE

    :param mask: uint8 or int np.array mask with len(shape) == 2 like grayscale image
    :return: list of ints in RLE format
    """
    assert len(mask.shape) == 2, 'mask must be 2D np.array'
    assert mask.dtype == np.uint8 or mask.dtype == int, 'mask must be uint8 or int'

    array = mask.ravel()
    array = np.repeat(array, 4)  # must be 4 channels
    rle = encode_rle(array)
    return rle

def set_page():
    pass


def call_lsf(description, config_x, interfaces, user, state,):
    st_label_studio(description, config_x, interfaces, user, state,)

def handevent(event, containerstep2):
    if event is None or ('Annotation' not in event['event']):
        return
    
    # 将JSON数据转换为字符串
    json_str = json.dumps(event)

    # 将JSON字符串转换为字节
    json_bytes = json_str.encode('utf-8')

    containerstep2.download_button(
        label="annotation files",
        data=json_bytes,
        file_name="data.json",
        mime='application/json'
    )
    
