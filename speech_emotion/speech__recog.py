import numpy as np
import librosa
from moviepy import VideoFileClip
from keras.models import load_model # type: ignore
import pickle
import streamlit as st
import json
import os
from datetime import datetime


def speech_recog(video_path,employee_id):
     video = VideoFileClip(video_path)
     audio_path = "output_audio.mp3"

        
     if video.audio is None:
        st.error("The video file does not contain an audio track.")
        return  

     video.audio.write_audiofile(audio_path)
    
     
# Load the model
     
     loaded_model = load_model('C:\\Users\\sarth\\OneDrive\\Documents\\python code\\zidio_project_1\\speech_emotion\\my_model.keras')
     with open('C:\\Users\\sarth\\OneDrive\\Documents\\python code\\zidio_project_1\\speech_emotion\\encoder.pkl', 'rb') as f:
          enc = pickle.load(f)

# Function to extract MFCC features
     def extract_mfcc(filename):
         y, sr = librosa.load(filename, duration=3, offset=0.5)
         mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
         return mfcc

# Load a new audio file for prediction
     new_audio_path = audio_path
     mfcc_features = extract_mfcc(new_audio_path)

# Prepare the features for prediction
     mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
     mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # Add channel dimension

# Make a prediction
     predictions = loaded_model.predict(mfcc_features)
     predicted_class = np.argmax(predictions, axis=1)

# Decode the predicted class back to the emotion label
     emotion_labels = enc.categories_[0]  # Assuming you have the encoder from training
     predicted_emotion = emotion_labels[predicted_class[0]]
     print(f'The predicted emotion is: {predicted_emotion}')
     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     individual_file = f"emotion_results_{employee_id}_{timestamp}.json"
     with open(individual_file, "w") as f:
         json.dump({"predicted_emotion": predicted_emotion}, f, indent=4)
         
     central_file = "all_emotion_results(speech).json"
     try:
         with open(central_file, "r") as f:
             all_results = json.load(f)
     except FileNotFoundError:
         all_results = {}  
     all_results[employee_id] = all_results.get(employee_id, [])
     all_results[employee_id].append({
     "timestamp": timestamp,
     "results": predicted_emotion  
     })
     with open(central_file, "w") as f:
         json.dump(all_results, f, indent=4)
     st.success(f"Speech evaluation complete and results have been stored in database")
     try:
         os.remove(individual_file)
         os.remove(audio_path)
         
     except Exception as e:
         st.error(f"error: {e}")
     st.session_state.uploaded_today = True









































# U
