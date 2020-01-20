from collections import deque
from imutils.video import VideoStream
from colorama import init
from termcolor import colored
import argparse
import imutils
import numpy as np
import time
import cv2
import os
import shutil
import skvideo
import skimage
import errno

import serial #pip install pyserial
import matplotlib.pyplot as plt
import scipy.integrate as it
import scipy.signal as sig
import math
import pandas as pd

skvideo.setFFmpegPath('C:\\Users\\chase\\Downloads\\ffmpeg\\ffmpeg\\bin')

import skvideo.io



#Define and Construct the Command Line Argument Parser (Either Use Pre-Filmed Video or Real-Time Video)
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="Insert Video Title Here")
parser.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = parser.parse_args()


G = -9.81

#Step 1: Pre-pair the HC-05 device to computer (The HC-05 has a default password when pairing)
#Step 2: Bound the HC-05 device to a certain COM port (Outgoing COM port number can be found in bluetooth settings menu on computer or COM port can be defined in Arduino IDE)

#Initialize Serial Object (Correct COM port name can be found in Device Manager)
message = serial.Serial('COM6', 38400, timeout = 1)
print("Connected")

#Hold Program For Serial Communication to be Established
time.sleep(5)

lift_num = 0


#Message being sent to HC-05 to tell microcontroller that we are pulling data from IMU
Lift = input("NEW lift? (Yes or No): ")
while(Lift == 'Yes'):

    # Define Lower and Upper Bounds For Colors in BGR Color Space Based on User Input of Weight Plate Color.
    init()
    RedText = colored("Red", 'red')
    BlueText = colored("Blue", 'blue')
    YellowText = colored("Yellow", 'yellow')
    GreenText = colored("Green", 'green')
    Red = "Red"
    Blue = "Blue"
    Yellow = "Yellow"
    Green = "Green"
    Black = "Black"
    RedDemo = "RedDemo"
    print(
        "Enter the Color of your weight plate (" + RedText + ", " + YellowText + ", " + GreenText + ", " + BlueText + ") " + ":",
        end=" ")
    BumperColor = input()
    if (BumperColor == Red):
        LowerBound = (170, 0, 20)
        UpperBound = (179, 255, 255)
    elif (BumperColor == Blue):
        LowerBound = (100, 50, 20)
        UpperBound = (140, 255, 255)
    elif (BumperColor == Yellow):
        LowerBound = (15, 100, 20)
        UpperBound = (35, 255, 255)
    elif (BumperColor == Green):
        LowerBound = (65, 60, 20)
        UpperBound = (80, 255, 255)
    elif (BumperColor == Black):
        LowerBound = (0, 0, 0)
        UpperBound = (180, 255, 30)
    elif (BumperColor == RedDemo):
        LowerBound = (0, 200, 20)
        UpperBound = (10, 255, 255)
    # Define Deque Data Structure (Store X,Y Points of Tracked Object).
    TrackedPoints = deque(maxlen=64)

    # Variable To Track Whether Using Pre-Filmed or Live-Stream Video
    Flag = 0

    # If Using Pre-Filmed Video, Grab Reference to it.
    if args.video:
        preloadedVideo = args.video
        videoStream = cv2.VideoCapture(preloadedVideo)

    # If Using Real-Time Video, Grab Reference to Webcam
    else:
        videoStream = VideoStream(src=0).start()
        Flag = 1

    # Wait For Webcam/Video File
    time.sleep(2.0)

    # Keep Track of Image Frames For Saved Video File
    frameArray = []
    count = 0
    try:
        os.mkdir("Frames")
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    try:
        os.mkdir("LiftSense Analysis")
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    #Create arrays to hold data from IMU
    data_write = []
    AccX = []
    AccY = []
    AccZ = []
    Pitch = []
    Roll = []
    Heading = []

    #Write 1 to Microcontroller to tell it to start polling IMU for data
    message.write(bytes('1', 'utf-8'))
    print("Message Successfully Sent")
    print("Writing data")
    print("Press Ctrl+C when lift finished")


    try:
        #Start Looping To Capture Video Stream and Perform Localization & Filtering
        while True:

            #Keep Track of Number of Frames
            count = count + 1

            #If Using Pre-Filmed Video, Capture Video Stream
            if(Flag == 0):
                ret, frame = videoStream.read()
                frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)

            #If Using Real-Time Video, Capture Video Stream
            else:
                frame = videoStream.read()

            #If No Frames Left in Pre-Filmed Video Then Exit
            if frame is None:
                break

            #Resize Frames For Faster FPS
            frame = imutils.resize(frame, width=600)

            #Convert RGB Color Space to HSV Color Space
            blurred = cv2.GaussianBlur(frame, (11,11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            #Perform Localization of Object and Filtering of Object
            mask = cv2.inRange(hsv, LowerBound, UpperBound)
            mask = cv2.erode(mask, None, iterations = 2)
            mask = cv2.dilate(mask, None, iterations = 2)

            #Compute the Outline/Contours of Object
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            #Determine The Center/Centroid of Object
            center = None
            if len(contours) > 0:
                centroid = max(contours, key = cv2.contourArea)
                ((x,y), radius) = cv2.minEnclosingCircle(centroid)
                moment = cv2.moments(centroid)
                center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
                if radius > 10:
                    #Draw the Contour and Centroid on the Video Stream
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            #Add Center/Centroid of Object to Data Structure To Keep Track of Position
            TrackedPoints.append(center)


            #Loop Through Positions of Object and Draw Line Connecting New Position to Previous Position
            for i in range(1, len(TrackedPoints)):
                if TrackedPoints[i - 1] is None or TrackedPoints[i] is None:
                    continue
                thickness = int(np.sqrt(args.buffer / float(i+1)) * 2.5)
                cv2.line(frame, TrackedPoints[i-1], TrackedPoints[i], (0, 0, 255), thickness)

            #Display Frame
            cv2.imshow("Frame", frame)

            #Save Frames and Append To Frame Array
            skimage.io.imsave("Frames/image" + str(count) + ".jpg", frame)
            frameArray.append(cv2.imread("Frames/image" + str(count) + ".jpg"))

            #Terminate Video Stream Using 'q' Key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        #Write 0 to Microcontroller to stop it from polling IMU
        message.write(bytes('0', 'utf-8'))
        pass


    #Stop Live-Stream if Not Using Pre-Recorded Video
    if (Flag == 1):
        videoStream.stop()

    #Stop Camera if Using Pre-Recorded Video
    if (Flag == 0):
        videoStream.release()

    #Delete Directory of Frames
    shutil.rmtree("Frames")

    #Write Video File with Feedback
    skvideo.io.vwrite("LiftSense Analysis\lift" + str(lift_num) + ".avi",frameArray)

    #Close All Windows
    cv2.destroyAllWindows()

    while(message.in_waiting):
        data = message.readline()
        data = data.decode('utf-8')
        data_write.append(data.rstrip())

    #Collect data in arrays using delimiter value
    for i in data_write:
        if (i != ''):
            if (i[0] == 'x'):
                AccX.append(float(i.strip('x')) * 9.81)
            elif (i[0] == 'y'):
                AccY.append(float(i.strip('y')) * 9.81)
            elif (i[0] == 'z'):
                AccZ.append(float(i.strip('z')) * 9.81)
            elif (i[0] == 'p'):
                Pitch.append(float(i.strip('p')))
            elif (i[0] == 'r'):
                Roll.append(float(i.strip('r')))
            elif (i[0] == 'h'):
                Heading.append(float(i.strip('h')))

    #Convert arrays to pd series for window averaging
    AccZ_s = pd.Series(AccZ)
    Roll_s = pd.Series(Roll)
    Pitch_s = pd.Series(Pitch)
    Heading_s = pd.Series(Heading)

    #Perfrom window average with window size equat to 5 data points
    smooth_accz = AccZ_s.rolling(5).mean()
    smooth_roll = Roll_s.rolling(5).mean()
    smooth_pitch = Pitch_s.rolling(5).mean()
    smooth_heading = Heading_s.rolling(5).mean()

    #Remove nan values generated from window averaging (first 4 values of array)
    smooth_accz = smooth_accz[np.logical_not(np.isnan(smooth_accz))]
    smooth_roll = smooth_roll[np.logical_not(np.isnan(smooth_roll))]
    smooth_pitch = smooth_pitch[np.logical_not(np.isnan(smooth_pitch))]
    smooth_heading = smooth_heading[np.logical_not(np.isnan(smooth_heading))]

    #Convert pd seris back to np arrays
    smooth_accz = np.array(smooth_accz)
    smooth_roll = np.array(smooth_roll)
    smooth_pitch = np.array(smooth_pitch)
    smooth_heading = np.array(smooth_heading)

    #Get length of smoothed arrays
    sa = len(smooth_accz)
    sr = len(smooth_roll)
    sp = len(smooth_pitch)
    sh = len(smooth_heading)

    #Get minimum length
    smallest = min(sa, sr, sp, sh)

    #Check each arrays length with the shortest array. If its bigger than smallest remove
    #the last element. This prevents errors in the next section
    if (sa - smallest > 0):
        smooth_accz = smooth_accz[:-1]
    if(sr - smallest > 0):
        smooth_roll = smooth_roll[:-1]
    if(sp - smallest > 0):
        smooth_pitch = smooth_pitch[:-1]
    if(sh - smallest > 0):
        smooth_heading = smooth_heading[:-1]

    #Calculate rotation factor using roll and pitch. Use that factor to remove effect of gravity in
    #Z direction acceleration.
    for x in range(len(smooth_accz)):
        r = smooth_roll[x]
        p = smooth_pitch[x]
        p = p * (math.pi / 180.0)
        r = r * (math.pi / 180.0)
        factor = G * math.cos(p)*math.cos(r)

        if ~np.isnan(AccZ[x]) or ~np.isnan(smooth_accz):
            AccZ[x] = AccZ[x] + factor
            smooth_accz[x] = smooth_accz[x] + factor

    #Perform integration of acceleration data in the z direction
    Smooth_Accz_int = it.cumtrapz(smooth_accz, dx=0.1, initial=0)

    #Detrend data
    accz_int_detrend = sig.detrend(Smooth_Accz_int)
    accz_int_detrend_copy = sig.detrend(Smooth_Accz_int)

    #X values are take to be 0.1s apart, this is the sample rate within the microcontroller
    x = np.linspace(0, round(((len(Smooth_Accz_int)+4)*0.136),1), len(Smooth_Accz_int))

    #Get peak velocity value
    peak = max(accz_int_detrend_copy)
    peak = round(peak, 2)

    plt.figure(1)
    plt.plot(x, accz_int_detrend, 'b')

    if(peak < 1.0):
       plt.ylim(-1.0, 1.0)

    plt.title('Velocity vs. Time')

    #Display peak velocity in top right corner
    plt.text(0.7, 0.95, "Peak Velocity: " + str(peak) + " m/s", transform=plt.gcf().transFigure, fontsize=8)

    #Label axis
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')

    #Save the figure
    plt.savefig("LiftSense Analysis\Velocity" + str(lift_num) + ".png")
    print("Results saved in LiftSense Analysis")

    plt.clf()

    Lift = input("NEW lift? (Yes or No): ")
    lift_num = lift_num + 1

exit()
