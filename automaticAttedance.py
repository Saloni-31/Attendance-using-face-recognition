import tkinter as tk
from tkinter import *
import os, cv2
import shutil
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import tkinter.ttk as tkk
import tkinter.font as font

haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel/Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "StudentDetails/studentdetails.csv"
attendance_path = "Attendance"

# Function for choosing subject and filling attendance
def subjectChoose(text_to_speech):
    def FillAttendance():
        sub = tx.get().strip()
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
            return
        Subject = sub
        path = os.path.join(attendance_path, Subject)
        
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        
        now = time.time()
        future = now + 20
        if not os.path.isfile(trainimagelabel_path):
            e = "Model not found, please train the model"
            Notifica.configure(
                text=e,
                bg="black",
                fg="yellow",
                width=33,
                font=("times", 15, "bold")
            )
            Notifica.place(x=20, y=250)
            text_to_speech(e)
            return

        # Face recognition and attendance recording
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainimagelabel_path)
        facecasCade = cv2.CascadeClassifier(haarcasecade_path)
        df = pd.read_csv(studentdetail_path)
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ["Enrollment", "Name"]
        attendance = pd.DataFrame(columns=col_names)
        
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = facecasCade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y : y + h, x : x + w])
                if conf < 70:
                    aa = df.loc[df["Enrollment"] == Id]["Name"].values
                    tt = str(Id) + "-" + str(aa)
                    attendance.loc[len(attendance)] = [Id, aa]
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 4)
                    cv2.putText(im, str(tt), (x + h, y), font, 1, (255, 255, 0), 4)
                else:
                    Id = "Unknown"
                    tt = str(Id)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                    cv2.putText(im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4)
            if time.time() > future:
                break
            attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
            cv2.imshow("Filling Attendance...", im)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Save attendance to CSV
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        fileName = f"{path}/{Subject}_{date}_{timeStamp}.csv"
        attendance.to_csv(fileName, index=False)

        # Notify the user
        m = "Attendance Filled Successfully for " + Subject
        Notifica.configure(
            text=m,
            bg="yellow",
            fg="black",
            width=33,
            relief=RIDGE,
            bd=5,
            font=("times", 15, "bold")
        )
        text_to_speech(m)
        Notifica.place(x=20, y=250)

        # Display attendance in GUI
        root = Tk()
        root.title(f"Attendance of {Subject}")
        root.configure(background="black")
        with open(fileName, newline="") as file:
            reader = csv.reader(file)
            for r, row in enumerate(reader):
                for c, col in enumerate(row):
                    label = tk.Label(
                        root,
                        width=10,
                        height=1,
                        fg="yellow",
                        font=("times", 15, "bold"),
                        bg="black",
                        text=col,
                        relief=tk.RIDGE
                    )
                    label.grid(row=r, column=c)
        root.mainloop()

    def Attf():
        sub = tx.get().strip()
        if sub == "":
            text_to_speech("Please enter the subject name!!!")
        else:
            os.startfile(os.path.join(attendance_path, sub))

    # GUI setup for subject entry
    subject = Tk()
    subject.title("Subject...")
    subject.geometry("580x320")
    subject.configure(background="black")
    
    # Notification Label
    Notifica = tk.Label(
        subject,
        text="Attendance filled Successfully",
        bg="yellow",
        fg="black",
        width=33,
        height=2,
        font=("times", 15, "bold")
    )

    # Entry for subject name
    tx = tk.Entry(
        subject,
        width=15,
        bd=5,
        bg="black",
        fg="yellow",
        relief=RIDGE,
        font=("times", 30, "bold")
    )
    tx.place(x=190, y=100)

    # Fill Attendance button
    fill_a = tk.Button(
        subject,
        text="Fill Attendance",
        command=FillAttendance,
        bd=7,
        font=("times new roman", 15),
        bg="pink",
        fg="white",
        height=2,
        width=12,
        relief=RIDGE
    )
    fill_a.place(x=195, y=170)

    # Check Sheets button
    attf = tk.Button(
        subject,
        text="Check Sheets",
        command=Attf,
        bd=7,
        font=("times new roman", 15),
        bg="pink",
        fg="white",
        height=2,
        width=10,
        relief=RIDGE
    )
    attf.place(x=360, y=170)
    
    subject.mainloop()