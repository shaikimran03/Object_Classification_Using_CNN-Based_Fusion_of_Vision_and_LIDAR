from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
from tkinter import ttk
import os
import tensorflow as tf
import os
import sys
from keras.models import model_from_json
import pickle
from keras.applications.inception_v3 import InceptionV3

main = tkinter.Tk()
main.title("Object Classification Using CNN-Based Fusion of Vision and LIDAR in Autonomous Vehicle Environment")
main.geometry("1200x1200")


global filename
global lidar
global inception_model

COLORS = np.random.uniform(0, 255, size=(21, 3))

def getClass(idx):
    if idx == 2:
        return "Cyclist"
    if idx == 7:
        return "Car"
    if idx == 15:
        return "Pedestrian"
    if idx == 6:
        return "Truck"
    else:
        return "Others"

def upload():
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    pathlabel.config(text=filename+" dataset loaded")

def loadLidar():
    global lidar
    lidar = cv2.dnn.readNetFromCaffe("model/alexnet.txt","model/alexnet.caffemodel")
    pathlabel.config(text="LIDAR CNN Model loaded")

def lidarClassification():
    global filename, lidar
    row = 50
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    text.insert(END,str(filename)+" loaded\n")
    pathlabel.config(text=str(filename)+" loaded")
    image_np = cv2.imread(filename)
    image_np = cv2.resize(image_np,(800,500))
    (h, w) = image_np.shape[:2]
    classification = tf.Graph()
    with classification.as_default():
        od_graphDef = tf.GraphDef()
        with tf.gfile.GFile('model/frozen_inference_graph.pb', 'rb') as file:
            serializedGraph = file.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')
    with classification.as_default():
        with tf.Session(graph=classification) as sess:
            blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)),0.007843, (300, 300), 127.5)
            lidar.setInput(blob)
            detections = lidar.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    print(confidence * 100)
                    if (confidence * 100) > 70:
                        label = "{}: {:.2f}%".format(getClass(idx),confidence * 100)
                        cv2.putText(image_np, label, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                        row = row + 30
                        text.insert(END,"Detected & Classified Objects: "+getClass(idx)+"\n")
                    if (confidence * 100) > 50:
                        cv2.rectangle(image_np, (startX, startY), (endX, endY),COLORS[idx], 2)
    text.update_idletasks()
    cv2.imshow('LIDAR Object Classification Output', image_np)
    cv2.waitKey(0)


def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('RGB-LIDAR Accuracy & Loss Graph')
    plt.show()

font = ('times', 14, 'bold')
title = Label(main, text='Object Classification Using CNN-Based Fusion of Vision and LIDAR in Autonomous Vehicle Environment')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload Kitti Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

lidarButton = Button(main, text="Load Alexnet LIDAR CNN Model", command=loadLidar)
lidarButton.place(x=50,y=150)
lidarButton.config(font=font1)

clsButton = Button(main, text="Run LIDAR Object Detection & Classification", command=lidarClassification)
clsButton.place(x=50,y=200)
clsButton.config(font=font1)

graphButton = Button(main, text="LIDAR Accuracy & Loss Graph", command=graph)
graphButton.place(x=480,y=200)
graphButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=480,y=100)

font1 = ('times', 12, 'bold')
text=Text(main,height=18,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
