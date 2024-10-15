import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from Home import face_rec

st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')

# inti registration Form
registration_form = face_rec.RegistrationForm()

# Step 1: Collect person name and Form
# form
person_name = st.text_input(label='Name', placeholder='First and Last Name')
role = st.selectbox(label='Select your Role', options=('Student', 'Teacher'))
department = st.selectbox(label='Selcet your Department', 
                          options=('Computer Engineering',
                                    'Software Engineering',
                                    'Computer Programming',
                                    'Civil Engineering',
                                    'Electronic and Electrical Engineering',
                                    'Architecture',
                                    'International Relations and Political Science',
                                    'International Law',
                                    'Business Administration',
                                    'Psychology',
                                    'Nursing',
                                    'Pharmacy'))


# Step 2: Collect Facial Embedding of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24')  # 3d array bgr
    reg_img, embedding = registration_form.get_embedding(img)

    # two step porcess
    # 1st save data into local computer txt
    if embedding is not None:
        with open('face_embedding.txt', mode='ab') as f:
            np.savetxt(f, embedding)

    return av.VideoFrame.from_ndarray(reg_img, format='bgr24')


webrtc_streamer(key='registration', video_frame_callback=video_callback_func)


# Step 3: Save the data in Redis Database


if st.button('Submit'):
    st.write(f'Person Name = ', person_name)
    st.write(f'Your role = ', role)
    st.write(f'Your Department = ', department)
    return_val = registration_form.save_data_in_redis_db(
        person_name, role, department)

    if return_val == True:
        st.success(f'{person_name} registered successfully')
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')

    elif return_val == 'file_false':
        st.error(
            'face_embedding.txt is not found. Please refresh the page and execute again.')
