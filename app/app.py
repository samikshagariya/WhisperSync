import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('WhisperSync')
    st.info('Final Year Project')

st.title('WhisperSync Lip Reading App')

data_path = os.path.join('data', 's1')
st.write(f"Looking for data at: {os.path.abspath(data_path)}")
options = [f for f in os.listdir(data_path) if f.endswith('.mpg')]
selected_video = st.selectbox('Choose Video', options)

col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data', 's1', selected_video)
        output_path = 'test_video.mp4'
        cmd = f'ffmpeg -i {file_path} -vcodec libx264 {output_path} -y'
        status = os.system(cmd)
        if status == 0 and os.path.exists(output_path):
            video = open(output_path, 'rb')
            video_bytes = video.read()
            st.video(video_bytes)
        else:
            st.error(f'Error: Failed to convert video. Status code: {status}')

    with col2:
        file_path = os.path.join('data', 's1', selected_video)
        try:
            video, annotations = load_data(tf.convert_to_tensor(file_path))
        except FileNotFoundError as e:
            st.error(f"Required file missing: {e}")
            st.stop()

        st.info('This is the output of the machine learning as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        input_len = tf.constant([yhat.shape[1]], dtype=tf.int32)
        decoded_dense, _ = tf.keras.backend.ctc_decode(yhat, input_length=input_len, greedy=True)
        decoded = decoded_dense[0].numpy()

        st.text(decoded[0])

        st.info('This is the decoded text')
        filtered_tokens = decoded[0][decoded[0] != -1]
        converted_preds = tf.strings.reduce_join(num_to_char(filtered_tokens)).numpy().decode('utf-8')
        st.text(converted_preds)
