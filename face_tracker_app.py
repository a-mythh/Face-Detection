import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


face_tracker = load_model('facetracker.h5')

cap = cv2.VideoCapture(0)

# change colour and thickness of box here
rect_properties = {
    'colour': (255, 255, 255),      # (B, G, R) -> 'cause opencv is crazy
    'thickness': 2
}

# Press 'q' to close the window

while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500, :]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    
    yhat = face_tracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        cv2.rectangle(frame, tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                            tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)), 
                            rect_properties['colour'], rect_properties['thickness'])
    
    cv2.imshow('Face Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()