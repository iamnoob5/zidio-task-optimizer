import cv2
from deepface import DeepFace
from collections import defaultdict
import streamlit as st
import json
from datetime import datetime
import os


def process_video(video_path,employee_id):
   faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   cap = cv2.VideoCapture(video_path)

   if not cap.isOpened():
       raise IOError("Could not open video file")


   emotion_sums = defaultdict(float)
   emotion_counts = defaultdict(int)

   while True:
       ret, frame = cap.read()
    
       if not ret:  
               print("End of video")
               break

    
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = faceCascade.detectMultiScale(gray, 1.1, 4)

   
       if len(faces) > 0:
           result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
           for res in result:
               for emotion, score in res['emotion'].items():
                   emotion_sums[emotion] += score  
                   emotion_counts[emotion] += 1   


   cap.release()
   average_emotions = {emotion: emotion_sums[emotion] / emotion_counts[emotion] for emotion in emotion_sums if emotion_counts[emotion] > 0}
   emotion_results = {key: float(value) for key, value in average_emotions.items()}
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   individual_file = f"emotion_results_{employee_id}_{timestamp}.json"

         
   with open(individual_file, "w") as f:
         json.dump(emotion_results, f, indent=4)

   central_file = "all_emotion_results(face).json"
   try:
         with open(central_file, "r") as f:
            all_results = json.load(f)
   except FileNotFoundError:
         all_results = {} 

   all_results[employee_id] = all_results.get(employee_id, [])
   all_results[employee_id].append({
         "timestamp": timestamp,
         "results": emotion_results
   })

   with open(central_file, "w") as f:
         json.dump(all_results, f, indent=4)
   st.success(f"Visual evaluation complete and results have been stored in database")

   try:
         os.remove(individual_file)

   except Exception as e:
         st.error(f"Error: {e}") 

   st.session_state.uploaded_today = True  

  