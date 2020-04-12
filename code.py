import sys
import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.createLBPHFaceRecognizer()
def pack_data_train():
    label3.pack()
    label4.pack()
    label5.pack()
    label6.pack()
    label_1.pack()
    label8.pack()

    label1.pack()
    E1.pack()
    label2.pack()
    E2.pack()

    button4.pack(side=BOTTOM)

def entry_data_train():
    naam = E1.get()
    lab = E2.get()
    return naam, lab

def pack_data_test():
    new_label.pack_forget()
    label3.pack()
    label4.pack()
    label5.pack()
    label6.pack()
    label_2.pack()
    label8.pack()

    label2.pack()
    E2.pack()
    button5.pack(side=BOTTOM)

def entry_data_test():
    lab = E2.get()
    return lab

def open_cam():

 c = 1
 name,f = entry_data_train()
 i = 0
 while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite("yalefaces/"+"subject"+str(f)+"."+str(name)+"."+str(c)+".jpg", gray[y:y+h, x:x+w])
        cv2.imwrite("trainingSet/"+"subject"+str(f)+"."+str(name)+"."+str(c)+".jpg", gray[y:y+h, x:x+w])
        c+=1
    cv2.imshow('img', img)
    i=i+1


    k = cv2.waitKey(400) & 0xFF
    if i > 15:
        break
 cap.release()
 cv2.destroyAllWindows()
 new_label.pack()
 E1.pack_forget()
 E2.pack_forget()
 button4.pack_forget()
 label_1.pack_forget()
 label1.pack_forget()
 label2.pack_forget()

def test_image():
  f = entry_data_test()
  while True:

      ret, img = cap.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          cv2.imwrite("yalefaces/"+"subject"+str(f)+"."+"sad"+".jpg", gray[y:y+h, x:x+w])
          cv2.imwrite("testSet/"+"subject"+str(f)+"."+"sad"+".jpg", gray[y:y+h, x:x+w])

      cv2.imshow('img', img)
      k = cv2.waitKey(200) & 0xFF
      if k == 27:
         break

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    from PIL import Image
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad.jpg')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        # Detect the face in the image
        faces = face_cascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

def recognition_test():
    from PIL import Image
    path = './yalefaces'

    images, labels = get_images_and_labels(path)
    cv2.destroyAllWindows()

    # Perform the training
    recognizer.train(images, np.array(labels))

    # Append the images with the extension .sad into image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad.jpg')]
    for image_path in image_paths:
       predict_image_pil = Image.open(image_path).convert('L')
       predict_image = np.array(predict_image_pil, 'uint8')
       faces = face_cascade.detectMultiScale(predict_image)

       for (x, y, w, h) in faces:
          nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
          nbr_actual_test = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
          if conf < 52:
             image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad.jpg')]
             for image_path in image_paths:
               nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
               if nbr_predicted == nbr_actual:
                 name_image = (os.path.split(image_path)[1].split(".")[1])
                 print "{} is Recognized as {}".format(nbr_actual_test, name_image)
                 break




          else:
             print "{} is Unknown because confidence is {}".format(nbr_actual_test, conf)

       cv2.imshow("Recognizing face", predict_image[y: y + h, x: x + w])
       cv2.waitKey(2000)




from Tkinter import *
root = Tk()
root.geometry('500x500')
root.title('Face detection and recognition system')
button1 = Button(root, text='Add faces to training set', command=pack_data_train, height=2, width=20)
button1.pack()
button2 = Button(root, text="Exit", command=exit, height=2, width=20)
button2.pack()
button3 = Button(root, text="Open camera for test", command=pack_data_test, height=2, width=20)
button3.pack()
button10 = Button(root, text="Recognize test images", command=recognition_test, height=2, width=20)
button10.pack()

label3 = Label(root, text="")
label4 = Label(root, text="")
label5 = Label(root, text="")
label6 = Label(root, text="")
label_1 = Label(root, text="Provide name of the person and label you like, and then click Submit")
label_2 = Label(root, text="Provide specific label and then click Submit")
label8 = Label(root, text="")
new_label = Label(root, text="Photos have been added to the training set.")

label1 = Label(root, text="Name of the person")
E1 = Entry(root, bd=5)
label2 = Label(root, text="Label")
E2 = Entry(root, bd=5)
button4 = Button(root, text="Submit", command=open_cam)
button5 = Button(root, text="Submit", command=test_image)

def opentraining():
    os.startfile('trainingSet')
def opentest():
    os.startfile('testSet')
def opendatabase():
    os.startfile('yalefaces')

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Training Set", command=opentraining)
filemenu.add_command(label="Test Set", command=opentest)
filemenu.add_command(label="Database", command=opendatabase)
menubar.add_cascade(label="Menu", menu=filemenu)
root.config(menu=menubar)
root.mainloop()


