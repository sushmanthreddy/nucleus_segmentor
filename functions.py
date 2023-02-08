import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import joblib

import numpy as np
import cv2
import onnxruntime as ort
import imutils
# import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def nucleus_segmentation():
    selected_box2 = st.sidebar.selectbox(
    'Choose Example Input',
    ('Example_1.png','Example_2.png')
    )

    st.title('Nucleus Segmentation')
    instructions = """
        Segment Nucleii from fluorescence microscopy imagery data (C. elegans embryo) \n
        Either upload your own image or select from the sidebar to get a preconfigured image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.text(instructions)
    file = st.file_uploader('Upload an image or choose an example')
    example_image = Image.open('./images/nucleus_segmentation_examples/'+selected_box2)
    threshold = st.sidebar.slider("Select Threshold (Applied on model output)", 0.0, 1.0, 0.1)
    col1, col2= st.beta_columns(2)

    if file:
        input = Image.open(file)
        fig1 = px.imshow(input, binary_string=True, labels=dict(x="Input Image"))
        fig1.update(layout_coloraxis_showscale=False)
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col1.plotly_chart(fig1, use_container_width=True)
    else:
        input = example_image
        fig1 = px.imshow(input, binary_string=True, labels=dict(x="Input Image"))
        fig1.update(layout_coloraxis_showscale=False)
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col1.plotly_chart(fig1, use_container_width=True)

    pressed = st.button('Run')
    if pressed:
        st.empty()
        fig2 = px.imshow(onnx_segment_nucleus(np.array(input), threshold), binary_string=True, labels=dict(x="Segmentation Map"))
        fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col2.plotly_chart(fig2, use_container_width=True)


def onnx_segment_nucleus(input_image, threshold):
    ort_session = ort.InferenceSession('onnx_models/nucleus_segmentor.onnx')
    img = Image.fromarray(np.uint8(input_image))
    resized = img.resize((256, 256), Image.NEAREST)
    img_unsqueeze = expand_dims_twice(resized)
    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')}) 
    binarized = 1.0 * (onnx_outputs[0][0][0] > threshold)
    resized_ret = Image.fromarray(binarized.astype(np.uint8) ).resize((708, 512), Image.NEAREST)#.convert("L")
    return(resized_ret)


def expand_dims_twice(arr):
    norm=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    ret = np.expand_dims(np.expand_dims(norm, axis=0), axis=0)
    return(ret)