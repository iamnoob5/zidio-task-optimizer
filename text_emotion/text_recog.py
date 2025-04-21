from moviepy import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
import joblib
import streamlit as st
import json
import os
from datetime import datetime

def text_recog(video_path,employee_id):
    
     video = VideoFileClip(video_path)
     audio_path = "output_audio.mp3"

        
     if video.audio is None:
        st.error("The video file does not contain an audio track.")
        return  

     video.audio.write_audiofile(audio_path)

     recognizer = sr.Recognizer()
     audio = AudioSegment.from_file(audio_path)
     audio.export(audio_path, format="wav")  # Ensure it's in WAV format for compatibility

     with sr.AudioFile(audio_path) as source:
         print("Processing audio...")
         audio_data = recognizer.record(source)

     try:
            # Convert speech to text
         transcription = recognizer.recognize_google(audio_data)
         print("Transcription:\n", transcription)

    # Save transcription to a file
         with open("transcription.txt", "w") as f:
             f.write(transcription)
         print("Transcription saved to 'transcription.txt'")
     except sr.UnknownValueError:
         print("Google Speech Recognition could not understand the audio.")
     except sr.RequestError as e:
         print(f"Could not request results from Google Speech Recognition; {e}")

     pipe_lr = joblib.load(open("C:\\Users\\sarth\\OneDrive\\Documents\\python code\\zidio_project_1\\text_emotion\\text_emo_model.pkl", "rb"))    
     def predict_emotions(docx):
         results = pipe_lr.predict([docx])
         return results[0]
     def get_prediction_proba(docx):
         results = pipe_lr.predict_proba([docx])
         return results
     with open("transcription.txt", "r") as file:
       file_content = file.read()
     prediction = predict_emotions(file_content)
     probability = get_prediction_proba(file_content)

     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     individual_file = f"emotion_results_{employee_id}_{timestamp}.json"
     emotion_results = {
             "predicted_emotion": prediction,
             "probabilities": probability.tolist()  # Convert to list for JSON serialization
     }
     with open(individual_file, "w") as f:
         json.dump(emotion_results, f, indent=4)
   
     central_file = "all_emotion_results(text).json"
     try:
        with open(central_file, "r") as f:
            all_results = json.load(f)
     except FileNotFoundError:
          all_results = {}  # Initialize if the file doesn't exist
     all_results[employee_id] = all_results.get(employee_id, [])
     all_results[employee_id].append({
      "timestamp": timestamp,
      "results": emotion_results  # Store the structured results
     })

     with open(central_file, "w") as f:
         json.dump(all_results, f, indent=4)
     st.success(f"Text evaluation complete and results have been stored in database")
     try:
         os.remove(individual_file)
         os.remove('transcription.txt')
     except Exception as e:
         st.error(f"error: {e}")
     st.session_state.uploaded_today = True
























