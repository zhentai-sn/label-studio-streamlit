'''
Author: zhentai-SMU 1753212182@qq.com
Date: 2024-08-16 16:12:11
LastEditors: zhentai-SMU 1753212182@qq.com
LastEditTime: 2024-08-16 16:25:31
FilePath: /glo_dmu/app/frontend/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import streamlit.components.v1 as components
_component_func = components.declare_component(
    name='streamlit_label_studio_frontend',
    path='./app/frontend'
)

# Public function for the package wrapping the caller to the frontend code
def st_label_studio(description, config, interfaces, user, task, key='label_studio_frontend', height=1000):
    component_value = _component_func(
        description=description, config=config, 
        interfaces=interfaces, user=user,
        task=task, height=height,
        key=key
    )
    return component_value
