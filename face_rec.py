import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os


import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# time
import time
from datetime import datetime

# Connect to Redis Client (database)
#  WSWMPcEJlDv84LtjvlQdGr5fnDQwYVyQ
#  redis-10190.c8.us-east-1-4.ec2.cloud.redislabs.com:10190

hostname = 'redis-10190.c8.us-east-1-4.ec2.cloud.redislabs.com'
portnumber = 10190
password = 'WSWMPcEJlDv84LtjvlQdGr5fnDQwYVyQ'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)


# retrive Data from Database
# 'academy:register'
# name = 'academy:register'

def retrive_data(name):
    retrive_dict = r.hgetall(name)

    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(
        lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))

    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'Facial Features']
    retrive_df[['Name', 'Role', 'Department']] = retrive_df['name_role'].apply(
        lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name', 'Role', 'Department', 'Facial Features']]


# configure face Analysis
faceapp = FaceAnalysis(name='buffalo_sc',
                       root="insightface_model/models",
                       providers=['CPUExecutionProvider'])

faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
# warning: don't set det_thresh < 0.3


# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_colunm, test_vector,
                        name_role=['Name', 'Role'], thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step 1: take teh dataframe (collection of data)
    dataframe = dataframe.copy()

    # step 2: Index face embedding form the dataframe nd convert into array
    X_list = dataframe[feature_colunm].tolist()
    x = np.asarray(X_list)

    # step 3: Calculate Cosine Similiarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['Cosine'] = similar_arr

    # step 4: Filter the data
    data_filter = dataframe.query(f'Cosine >= {thresh}')
    if len(data_filter) > 0:
        # step 5: get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['Cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]

    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role

# Real Time Prediction
# We need to save the log for every 1 minute


class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def saveLogs_redis(self):
        # step 1: create a logs dataframe
        dataframe = pd.DataFrame(self.logs)

        # step 2: drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name', inplace=True)

        # step 3: push data to redis database (list)
        # encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        # department_list = dataframe['department'].tolist()
        ctime_list = dataframe['current_time'].tolist()

        encoded_data = []

        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name} @ {role} @ {ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush("attendance:logs", *encoded_data)

        self.reset_dict()


###########################################################################################

    def face_prediction(self, test_image,
                        dataframe,
                        feature_colunm,
                        name_role=['Name', 'Role'],
                        thresh=0.5):

        # extra step: Find the time
        current_time = str(datetime.now())

        # step 1: take the test image and apply in insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        # step 2: use a for loop and extract each embedding and pass to ml_search_algorithm
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                           feature_colunm,
                                                           test_vector=embeddings,
                                                           name_role=name_role,
                                                           thresh=thresh)

            if person_name == 'Unknown':
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for recognized

            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color, 1)

            text_gen = person_name

            # Add text annotations to the image
            cv2.putText(test_copy, text_gen, (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(test_copy, current_time, (x1, y2+14),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            # self.logs['department'].append(person_department)
            self.logs['current_time'].append(current_time)

        # Convert BGR image to RGB for matplotlib
        # test_copy_rgb = cv2.cvtColor(test_copy, cv2.COLOR_BGR2RGB)

        # Display the image using plt
        # plt.figure(figsize=(16,24))
        # plt.imshow(test_copy_rgb)
        # plt.title('Test Image with Bounding Boxes')
        # plt.axis('off')  # Turn off the axis labels and ticks
        # plt.show()

        return test_copy


# Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        # get results from insightface model
        results = faceapp.get(frame, max_num=1)
        embeddings = None

        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # put text samples info
            text = f"samples =  {self.sample}"
            cv2.putText(frame, text, (x1, y1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)

            # facial features
            embeddings = res['embedding']

        return frame, embeddings

    def save_data_in_redis_db(self, name, role, department):

        # validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name} @ {role} @ {department}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        # if face_embeddings.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        # Step 1: Load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',
                             dtype=np.float32)  # flatten array

        # step 2: Convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # step 3: Cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # Step 4: Save this into redis database
        # redis hashes
        r.hset(name='academy:register', key=key, value=x_mean_bytes)

        # remove the file
        os.remove('face_embedding.txt')
        self.reset()

        return True
