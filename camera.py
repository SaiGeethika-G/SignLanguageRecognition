from imutils.video import VideoStream
from flask import Flask, render_template, request, Response
import pickle
app=Flask(__name__, template_folder='templates')
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

class VideoCamera(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        self.model = load_model('mp_hand_gesture')
        f = open('gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()
        print(self.classNames)

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret, frame = self.video.read()
        
        x, y, c = frame.shape

    	# Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    	# Get hand landmark prediction
        result = self.hands.process(framergb)

    	# print(result)
		
        className = ''

        if result.multi_hand_landmarks:
            landmarks=[]
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                	# print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

            	# Drawing landmarks on frames
                self.mpDraw.draw_landmarks(frame, handslms, self.mpHands.HAND_CONNECTIONS)

            	# Predict gesture
                prediction = self.model.predict([landmarks])
            	# print(prediction)
                classID = np.argmax(prediction)
                className = self.classNames[classID]

    	# show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


        ret, jpeg=cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        
    