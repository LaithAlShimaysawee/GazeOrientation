
#                      Gaze Orientation Estimation

The GazeTracking package has several methods to estimate gaze orientation, 
plot face contours, face meshs, irises contours, pupils centres. These methods are:

```
GazeEstimation.extract_face_landmarks(): This method returns arrays of face landmarks, irises landmarks as follow:
landmarks, face_landmarks : represent all face landmarks where (face_landmarks = landmarks.landmark)
irises_landmarks_pointer : represents the indexes of the irises landmarks inside the face_landmarks array
left_eye_landmarks_pointer: represents the indexes of the lefet eye landmarks inside the face_landmarks array
right_eye_landmarks_pointer: represents the indexes of the right eye landmarks inside the face_landmarks array

```
```
  GazeEstimation.get_left_pupil_centre(): This method returns the (x, y) point of the left eye pupil centre
```      
## GazeEstimation.get_right_pupil_centre(): 
        """This method returns the (x, y) point of the left eye pupil centre
        """
GazeEstimation.plot_face_mesh():    
        """This method returns the input image with face mesh plotted over it
        """
GazeEstimation.plot_face_contours():    
        """This method returns the input image with face contours plotted over it
        """ 
GazeEstimation.plot_irises_landmarks():    
        """This method returns the input image with irises contours plotted over it
        """ 
GazeEstimation.plot_pupils_centres():    
        """This method returns the input image with pupils centres plotted over it
        """
GazeEstimation.plot_plus(image, x, y):       
        """This method returns the input image with (+) symbol plotted over the given (x, y) point
        """
GazeEstimation.write_pupils_centres():
        """This method returns a black image with centre points of left and right pupils written over it
        """
GazeEstimation.plot_eyes_contours():    
        """This method returns the input image with both eyes contours plotted over it
        """
GazeEstimation.get_eyes_boundingbox():    
        """This method returns points of the bounding boxes for both eyes as follow:
        left_eye  bounding box    returned as [Ymin, Ymax, Xmin, Xmax]
        right_eye bounding box    returned as [Ymin, Ymax, Xmin, Xmax]
        """ 
GazeEstimation.horizontal_vertical_blinking_gaze_ratios():
            """Returns horizontal eyes ratio, vertical eyes ratio, left and right eyes blink ratios
            Each ratio is a number between 0.0 and 1.0 that indicates the
            horizontal, vertical direction of the gaze and the blinking ratio of each eye as follow:
            
            horizontal ratio: extreme right direction (= 0.0), centre (= 0.5), extreme left direction (= 1.0)
            vertical ratio: extreme top direction (= 0.0), centre (= 0.5), extreme bottom direction (= 1.0)
            left/right eyes blink ratios: blinknig (<= 0.3), open (> 0.3)
            """
GazeEstimation.estimate_gaze_direction():
        """This method returns a black image with text indicating gaze direction written over it
        """
GazeEstimation.plot_gaze_direction():
        """This method returns a black image with text indicating gaze direction written over it.
           It also returns a visualisation of the gaze direction of both eyes as two cirles 
           with arrows in side them pointong to the gize direction, or fully coloured to indicate eyes blinking.
        """
GazeEstimation.pol2cart(radius, angle):
        """ This method convert polar cordinates (radius, angle) to cartesian (x, y)
        """


#                      Installation of the package

on Command promt or powershell or Anaconda powershell
pip install git+https://github.com/LaithAlShimaysawee/GazeOrientation.git




#                      Demo1 example (main.py)

This demo example shows how to use the package to perform gaze tracking and orientation estimation over a webcam stream



#                      Demo2 example (main_Flask_APP.py)

This demo example shows how to use the package to build an APP to perform gaze tracking and orientation estimation over a webcam stream. This demo requires Flask library 

After running the APP using Command promt or powershell or Anaconda powershell, copy-paste http://127.0.0.1:5000/ into your favorite internet browser and it should be working.



#       credits to webistes(githubs, blogs, etc) that helped to complete this project


https://google.github.io/mediapipe/solutions/face_mesh.html

https://kh-monib.medium.com/title-gaze-tracking-with-opencv-and-mediapipe-318ac0c9c2c3

https://github.com/antoinelame/GazeTracking

https://towardsdatascience.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893

https://github.com/hemanth-nag/Camera_Flask_App

https://towardsdatascience.com/use-git-submodules-to-install-a-private-custom-python-package-in-a-docker-image-dd6b89b1ee7a



#                      My github profile


https://github.com/LaithAlShimaysawee






