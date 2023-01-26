import datetime, time
import os
import cv2
from flask import Flask, render_template, Response, request
from threading import Thread
from GazeOrientation.GazeTracking import GazeEstimation

global capture, rec_frame, gaze_direction, switch, face_contour, face_mesh, rec, out 
capture        = 0
gaze_direction = 0
face_contour   = 0
face_mesh      = 0
switch         = 1
rec            = 0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Instatiate flask app  
app = Flask(__name__, template_folder='./templates')

webcam = cv2.VideoCapture(0)

#------------------------------------------------------------------------------
def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

#------------------------------------------------------------------------------
def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = webcam.read() 
        if success:
            if(face_contour):
                estimate_gaze = GazeEstimation(frame)                
                frame         = estimate_gaze.plot_face_contours()
              
            if(face_mesh):
                estimate_gaze = GazeEstimation(frame)                
                frame         = estimate_gaze.plot_face_mesh()
       
            if(gaze_direction):
                estimate_gaze = GazeEstimation(frame)                
                frame1        = estimate_gaze.plot_pupils_centres()
                
                frame3        = estimate_gaze.plot_gaze_direction()
                frame         = cv2.flip(frame3, 1) +  frame1
                frame         = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            if(capture):
                capture = 0
                now     = datetime.datetime.now()
                p       = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, cv2.flip(frame,1))
            
            if(rec):
                rec_frame = frame
                frame     = cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame     = cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame       = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

#------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')
    
#------------------------------------------------------------------------------    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#------------------------------------------------------------------------------
@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,webcam
    if request.method == 'POST':
        if request.form.get('gaze_direction') == 'Estimate Gaze Direction':
            global gaze_direction
            gaze_direction = not gaze_direction
 
        elif  request.form.get('face_contour') == 'Show Face Contours':
            global face_contour
            face_contour = not face_contour

        elif  request.form.get('face_mesh') == 'Show Face mesh':
            global face_mesh
            face_mesh = not face_mesh

        elif  request.form.get('click') == 'Capture':
            global capture
            capture = 1               

        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch = 0
                webcam.release()
                cv2.destroyAllWindows()
                
            else:
                webcam = cv2.VideoCapture(0)
                switch=1
 
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec

            if(rec):
                now    = datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out    = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
 
            elif(rec==False):
                out.release()
                          
                 
    elif request.method =='GET':
        return render_template('index.html')
    return render_template('index.html')

#------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run()
    
webcam.release()
cv2.destroyAllWindows()     

