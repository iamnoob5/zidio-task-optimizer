import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from faceemotion.face_emotion import process_video
from speech_emotion.speech__recog import speech_recog
from text_emotion.text_recog import text_recog
import os
from datetime import datetime
import cv2


employee_credentials = {"employee1": "emp123", "employee2": "emp123","employee3": "emp123","employee4": "emp123","employee5": "emp123","employee6": "emp123","employee7": "emp123","employee8": "emp123","employee9": "emp123"}
management_credentials = {"manager1": "mgr123"}
employee_numbers = employee_credentials.keys()
TASKS_FILE = "tasks.json"


def load_tasks():
    try:
        with open("tasks.json", "r") as file:
            tasks = json.load(file)
        return tasks
    except FileNotFoundError:
        return []
def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f)
def create_task(name, complexity):
    return {"name": name, "complexity": complexity, "completed": False}
def delete_task(tasks, index):
    tasks.pop(index)
    save_tasks(tasks)
def save_tasks_emp(tasks):
    with open("tasks.json", "w") as file:
        json.dump(tasks, file, indent=4)


        
def display_tasks():
    st.title("Task Viewer")
    tasks = load_tasks()
    if tasks:
        st.subheader("Current Tasks")
        st.write("Here are the tasks assigned to you along with their complexity scores. In case of emotional distress, we recommend completing the first 3 urgently and to continue after finding some relaxation")
        for i, task in enumerate(tasks):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i + 1}. {task['name']} ({task['complexity']}) ")
            with col2:
                if st.button("Completed", key=f"delete_{i}"):
                    tasks.pop(i)
                    save_tasks_emp(tasks)
                    st.rerun()
    else:
        st.write("No tasks available.")    




def task_organizer():
    st.title("Task Organizer")
    tasks = load_tasks()
    with st.form("task_form"):
        task_name = st.text_input("Task Name")
        complexity_rating = st.text_input("Complexity Rating (Enter a number)", "")
        employee_number = st.selectbox("Assign to Employee", employee_numbers)
        submit_button = st.form_submit_button("Add Task")
        if submit_button and task_name:
            try:
                complexity_rating = float(complexity_rating) 
                new_task = {
                    "name": task_name,
                    "complexity": complexity_rating,
                    "employee_number": employee_number,
                }
                tasks.append(new_task)
                save_tasks(tasks)
                st.success(f"Task '{task_name}' assigned to {employee_number}!")
            except ValueError:
                st.error("Please enter a valid number for complexity rating.")
    if tasks:
        st.subheader("Current Tasks")
        for i, task in enumerate(tasks):
            st.write(f"{i + 1}. {task['name']} ({task['complexity']}) - Assigned to {task['employee_number']}")

        st.subheader("Delete Tasks")
        for i, task in enumerate(tasks):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{task['name']} ({task['complexity']})")
            with col2:
                if st.button("Delete", key=f"delete_{i}"):
                    tasks.pop(i)
                    save_tasks(tasks)
                    st.rerun()
    else:
        st.write("No tasks available. Please add a task.")





def employee_page():
    st.title("Employee Section")
    if st.button("Messages and Updates", key="messages_button"):
        st.session_state.show_updates = not st.session_state.get("show_updates", False)
    if st.session_state.get("show_updates", False):
        st.subheader("Messages and Updates")
        manage_automated_message()
    if st.button("Start Well-Being Evaluation for Today", key="evaluation_button"):
        st.session_state.show_evaluation = not st.session_state.get("show_evaluation", False)
    if st.session_state.get("show_evaluation", False):
        well_being_evaluation()
    if st.button("Tasks for this week", key="tasks_button"):
        st.session_state.show_tasks = not st.session_state.get("show_tasks", False)
    if st.session_state.get("show_tasks", False):
        display_tasks()    




def manage_automated_message():
    message_file_path = "automated_message.json"  
    try:
        with open(message_file_path, "r") as f:
            message_data = json.load(f)
        st.subheader("Message from manager")
        st.write(f" {message_data.get('message', 'No message available.')}")
        if st.button("Delete Message"):
            try:
                import os
                os.remove(message_file_path)
                st.success("Message has been successfully deleted.")
                st.rerun() 
            except Exception as e:
                st.error(f"An error occurred while deleting the message: {str(e)}")
    except FileNotFoundError:
        st.write("No message found.")    




def record_with_webcam(output_path):
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        st.error("Could not access the webcam. Please check your webcam.")
        return False
    st.info("Press 'q' to stop recording.")
    st.write("Recording from the webcam...")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))  
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            st.image(frame, channels="BGR")
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.success("Recording complete!")
    return True

         
         

def well_being_evaluation():
    if "username" not in st.session_state:
        st.error("Please log in to perform the well-being evaluation.")
        return
    employee_id = st.session_state["username"]    

    st.info("Give us a video recording of you, telling us about your day.")
    st.write("Upload a video or use the webcam to record one.")

   
    webcam_video_path = "webcam_video.avi" 
    if st.button("Record Video Using Webcam"):
        st.write("Opening webcam...")
        success = record_with_webcam(webcam_video_path)
        if success:
            st.write("Processing webcam video...")
            process_video(webcam_video_path, employee_id)
            speech_recog(webcam_video_path, employee_id)
            text_recog(webcam_video_path, employee_id)
            os.remove(webcam_video_path)  

    
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        video_path = f"temp_video.mp4"  

        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video uploaded successfully!")

        st.write("Processing video...")
        process_video(video_path, employee_id)
        speech_recog(video_path, employee_id)
        text_recog(video_path, employee_id)
        os.remove(video_path) 
    else:
        st.warning("Please upload a video to proceed.")
            





def login(credentials):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in credentials and credentials[username] == password:
            st.session_state["username"] = username  
            st.success(f"Welcome, {username}!")
            return True
        else:
            st.error("Invalid username or password. Please try again.")
            return False
    return False



def logout():
    if st.button("Logout"):
        st.session_state.pop("username", None)  
        st.success("You have been logged out.")




def send_message(employee_number, message):
    if "messages" not in st.session_state:
        st.session_state["messages"] = {}
    if employee_number not in st.session_state["messages"]:
        st.session_state["messages"][employee_number] = []
    st.session_state["messages"][employee_number].append(message) 





def analytics():
    st.title("Well being evaluation results and Alerting System")
   

    def load_json_file(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    
    
    face_res = "D:\\real documents\\python code\\zidio_project_1\\all_emotion_results(face).json"
    speech_res = "D:\\real documents\\python code\zidio_project_1\\all_emotion_results(speech).json"
    text_res = "D:\\real documents\\python code\\zidio_project_1\\all_emotion_results(text).json"

  
    data1 = load_json_file(face_res)
    data2 = load_json_file(speech_res)
    data3 = load_json_file(text_res)

    
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)

    def process_data(data):
      records = []
      for employee, entries in data.items():
        for entry in entries:
            timestamp = entry.get('timestamp', None)
            if not timestamp:
                continue
            results = entry.get('results', {})
            
            if isinstance(results, dict):
                record = {'employee': employee, 'timestamp': datetime.strptime(timestamp, "%Y%m%d_%H%M%S")}
                record.update(results)
            elif isinstance(results, str):
                record = {'employee': employee, 'timestamp': datetime.strptime(timestamp, "%Y%m%d_%H%M%S")}
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                for emotion in emotions:
                    record[emotion] = 1 if emotion == results else 0
            elif isinstance(results, dict) and 'predicted_emotion' in results:
                record = {'employee': employee, 'timestamp': datetime.strptime(timestamp, "%Y%m%d_%H%M%S")}
                probabilities = results.get('probabilities', [[]])[0]
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                for i, emotion in enumerate(emotions):
                    record[emotion] = probabilities[i] if i < len(probabilities) else 0
            
            records.append(record)
      return pd.DataFrame(records)


    df1 = process_data(data1)
    df2 = process_data(data2)
    df3 = process_data(data3)
    df = pd.concat([df1, df2, df3], ignore_index=True)
   
    def ensure_numeric(df):
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        for column in emotion_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df
    
    def preprocess_data(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])  
        df.set_index('timestamp', inplace=True)  
        df = ensure_numeric(df)  
        return df
    
    df = preprocess_data(df)


    def generate_threshold_alerts(df, threshold=25):
       if not isinstance(df.index, pd.DatetimeIndex):
           df.index = pd.to_datetime(df.index)
       df = df.sort_index()
       df['date'] = df.index.date
       alerts = []
       grouped = df.groupby('date')
       unique_dates = sorted(grouped.groups.keys())
 
       for i in range(len(unique_dates) - 1):
           day1 = unique_dates[i]
           day2 = unique_dates[i + 1]
           day1_data = grouped.get_group(day1)
           day2_data = grouped.get_group(day2)
        
          
           for employee in day1_data['employee'].unique():
               emp_data = day1_data[day1_data['employee'] == employee]
               if emp_data['angry'].max() >= threshold:
                   alerts.append(f"Alert: {employee} shows high anger levels ({emp_data['angry'].max():.1f}%) on {day1}.")
               if emp_data['sad'].max() >= threshold:
                   alerts.append(f"Alert: {employee} shows high sadness levels ({emp_data['sad'].max():.1f}%) on {day1}.")
               if emp_data['fear'].max() >= threshold:
                   alerts.append(f"Alert: {employee} shows high fear levels ({emp_data['fear'].max():.1f}%) on {day1}.")
               if emp_data['disgust'].max() >= threshold:
                   alerts.append(f"Alert: {employee} shows high disgust levels ({emp_data['disgust'].max():.1f}%) on {day1}.")
        
           
           negative_emotions = ['angry', 'disgust', 'fear', 'sad']
           day1_above_threshold = all(day1_data[emotion].max() >= threshold for emotion in negative_emotions)
           day2_above_threshold = all(day2_data[emotion].max() >= threshold for emotion in negative_emotions)
        
           if day1_above_threshold and day2_above_threshold:
               alerts.append(f"Alert: Multiple negative emotions detected above threshold ({threshold}) on {day1} and {day2}.")
    
       if alerts:
           st.header("Alerts")
           for alert in alerts:
               st.warning(alert)
        
          
           if 'alerts_shown' not in st.session_state:
               st.session_state.alerts_shown = True
               message = {
                    "message": "Our system detected that you are in emotional distress. Tasks have been reorganised for your comfort. If needed contact management for further support,couselling or stress management programs.", 
                    "timestamp": str(pd.Timestamp.now())
                }
               message_file_path = "automated_message.json"  
               with open(message_file_path, "w") as f:
                    json.dump(message, f, indent=4)
               st.success(f"Message saved to {message_file_path}.")
               try:
                   with open("tasks.json", "r") as f:
                       tasks = json.load(f)
                   tasks = sorted(tasks, key=lambda x: float(x['complexity']))
                   with open("tasks.json", "w") as f:
                       json.dump(tasks, f, indent=4)
                
                   st.success("Tasks have been successfully reorganized based on complexity rating.")
                   st.rerun() 
               except ValueError:
                   st.error("Complexity ratings must be numeric to sort tasks.")
               except FileNotFoundError:
                   st.error("tasks.json file not found. Ensure the file exists.")
               except Exception as e:
                   st.error(f"An unexpected error occurred: {str(e)}")
       else:
           st.write("No alerts to display.")
           
           if 'alerts_shown' in st.session_state:
               del st.session_state.alerts_shown
        
    st.header("Employee's Emotion Over Days")
    employee_names = df['employee'].unique()
    selected_employee = st.selectbox("Select Employee", employee_names)
    employee_data = df[df['employee'] == selected_employee]
    fig, ax = plt.subplots()
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']  
    for emotion, color in zip(emotions, colors):
        ax.scatter(employee_data.index, employee_data[emotion], label=emotion, color=color)
    plt.title(f"Emotions Over Days for {selected_employee}")
    plt.ylabel("Emotion Scores")
    plt.xlabel("Date")
    
   
    unique_dates = employee_data.index.unique()
    date_labels = [date.strftime('%d-%b') for date in unique_dates]
    
  
    ax.set_xticks(unique_dates)
    ax.set_xticklabels(date_labels, rotation=45)
    
    plt.legend()
    st.pyplot(fig)   
    generate_threshold_alerts(df)    
   
   
    st.header("Team's Emotion Over Days")
    team_size = 3
    total_employees = len(df['employee'].unique())
    if total_employees == 0:
         st.error("No employee data available to analyze.")
         return
    elif total_employees < team_size:
         total_teams = 1 
    else:
         total_teams = total_employees // team_size
    if total_employees < 9: 
         st.warning(f"Currently, you have data for only {total_employees} employee(s). Full analysis with 9 employees and 3 teams might not work as expected.")
     
    
    team_number = st.number_input("Select Team Number", min_value=1, max_value=total_teams, step=1)
  
    team_employees = [f'employee{i+1}' for i in range(min((team_number-1)*team_size, total_employees), min(team_number*team_size, total_employees))]

    team_data = df[df['employee'].isin(team_employees)]
    daily_team_data = team_data.resample('D').mean(numeric_only=True).fillna(0)
    fig, ax = plt.subplots()
    daily_team_data.mean()[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']].plot(kind='bar', ax=ax)
    plt.title(f"Average Team Emotions Over Days for Team {team_number}")
    plt.ylabel("Average Emotion Scores")
    plt.xlabel("Emotions")
    st.pyplot(fig)


    st.header("Team Mood Findings Compared to Other Teams")
    teams = df['employee'].unique()
 
    total_employees = len(df['employee'].unique())
    total_teams = total_employees // team_size


    if total_teams < 2:
        st.warning("Not enough data to compare team moods. At least two teams are required.")
        return

    team_moods = {}
    for i in range(0, len(df['employee'].unique()), team_size):
        team_name = f"Team {i // team_size + 1}"
        team_df = df[df['employee'].isin(teams[i:i+team_size])]
    
        if team_df.empty:
             st.warning(f"No data available for {team_name}. Skipping this team.")
             continue  
    
        
        team_df = team_df.reset_index()
        team_df = team_df.select_dtypes(include=['float', 'int']) 
        team_moods[team_name] = team_df.mean()


    if not team_moods:
        st.warning("No team data available for comparison.")
        return

    team_moods_df = pd.DataFrame(team_moods).T.fillna(0)
    if team_moods_df.empty:
        st.warning("Team mood comparison could not be generated due to insufficient data.")
        return

    fig, ax = plt.subplots()
    team_moods_df[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']].plot(kind='bar', ax=ax)
    plt.title("Team Mood Findings Compared to Other Teams")
    plt.ylabel("Average Emotion Scores")
    plt.xlabel("Teams")
    st.pyplot(fig)

    
  
def management_page():
    st.title("Management Section")
    st.write("Welcome to the management section. Content coming soon!")
    if st.button("Tasks for this week", key="tasks_button"):
        st.session_state.show_tasks = not st.session_state.get("show_tasks", False)
    if st.session_state.get("show_tasks", False):
        task_organizer() 
    if st.button("Mood Analytics", key="analytics_button"):
        st.session_state.show_analytics = not st.session_state.get("show_analytics", False)
    if st.session_state.get("show_analytics", False):
        analytics()    
    


st.title("Zidio AI-Powered Task Optimizer")
st.write("Please select a section to proceed:")

section = st.selectbox("Choose a section", ["Home", "Employee", "Management"])


if section == "Employee":
    st.subheader("Employee Login")
    if "username" in st.session_state:
        employee_page()
        logout()  
    else:
        if login(employee_credentials):
            employee_page()
elif section == "Management":
    st.subheader("Management Login")
    if "username" in st.session_state:
        management_page()
        logout() 
    else:
        if login(management_credentials):
            management_page()
else:
    st.write("Please make a selection from the dropdown above.")










                
    
       

       

      

    

       





           
            


    


    


   

        
       
    
  
        
       
         
        

        
            

           
            

    



        










    
    
    

    
    
    
    
   
    

    


    





    

   



        

       

    



  




















  

















    
















    

       



        


















