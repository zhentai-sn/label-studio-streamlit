'''
Description: 
Autor: zhentai
Date: 2024-07-10 20:46:46
LastEditTime: 2024-09-04 10:35:16
'''
import base64

import streamlit as st
import cv2
from PIL import Image
import json
import io
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os
from datetime import datetime
import json
import uuid

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) 

from frontend import st_label_studio
from utils.tools import *




def main():

    with st.sidebar:
        # 标题栏
        containertitle = st.container(border=False)
        containertitle.markdown('<div style="text-align: center; font-weight: bold; font-size: 24px;">Image Annotation Tool</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

        # 图像上传区
        containerstep1 = st.container(border=True)
        containerstep1.header("Upload  image")
        uploaded_files = containerstep1.file_uploader("Select one or more image files", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        st.markdown('<br>', unsafe_allow_html=True)

        # 标注文件下载区
        containerstep2 = st.container(border=True)
        containerstep2.header("Download")


    c1 = st.container(border=True) 
    c2 = st.container(border=True) 
    c3 = st.container(border=True)
    
    state = st.session_state
    description, config_x, interfaces, user, state.task = get_config()

    
    with c1:
        
        if uploaded_files:

            folder_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            folder_path_prefix = f"app/frontend/images/upload_imgs/{folder_stamp}"
            os.makedirs(folder_path_prefix, exist_ok=True)

            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)

                # 将上传的文件转换为图像
                image = Image.open(uploaded_file)
                img_array = np.array(image)

                # 将图像保存到临时文件
                temp_file_path = os.path.join(folder_path_prefix, uploaded_file.name)
                cv2.imwrite(temp_file_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                width, height = image.size
                print(width,height)

            state.task['data']['image'] = [os.path.join('images/upload_imgs', folder_stamp, x) for x in os.listdir(folder_path_prefix)] #只加载png

            print(state.task['data']['image'])
            event = st_label_studio(description, config_x, interfaces, user, state.task)
            handevent(event, containerstep2)


        # else:

        #     state.task['data']['image'] = ['images/examples/wallhaven-j3o7gq.jpg']
        #     event = st_label_studio(description, config_x, interfaces, user, state.task,)
        #     handevent(event, containerstep2)
            

if __name__ == '__main__':

    st.set_page_config(layout='wide')

    main()
