import cv2
from GazeOrientation.GazeTracking import GazeEstimation


#------------------------------------------------------------------------------
def main():

    webcam = cv2.VideoCapture(0)

    # set the setting of the webcam
    webcam.set(3, 1640)    # width
    webcam.set(4, 1420)    # height
    webcam.set(10, 100)    # brightness
    
    while webcam.isOpened():
        
        success, image = webcam.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        estimate_gaze = GazeEstimation(image)
        image1        = estimate_gaze.plot_pupils_centres()        
        text_image    = estimate_gaze.plot_gaze_direction()
        image         = cv2.flip(image1, 1) +  text_image
        image0        = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
      
           
        cv2.imshow('Gaze estimation project', image0)

       
    
    
        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()