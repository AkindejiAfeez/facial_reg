import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Predictions')
st.subheader('Real Time Face Prediction')


# STEP 1: Retrive the data from redis Database
with st.spinner('Retriving Data from redis DB...'):
    redis_face_db = face_rec.retrive_data(name='academy:register')
    st.dataframe(redis_face_db)
st.success("Data successfully retrieved from Redis")

# time
waitTime = 6  # time is in seconds
setTime = time.time()
realtimepred = face_rec.RealTimePred()  # real time prediction class


# STEP 2: Real Time Prediction
# streamlit webrtc

# callback function
def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24")  # a 3 dimesion numpy array

    # operations that you can perform on array
    # flipped = img[::-1,:,:]
    pred_img = realtimepred.face_prediction(img,
                                            redis_face_db,
                                            'Facial Features',
                                            ['Name', 'Role'],
                                            thresh=0.5)

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time()  # reset time
        print("Save Data to redis database")

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction",
                video_frame_callback=video_frame_callback)
