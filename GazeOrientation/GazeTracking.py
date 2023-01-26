import numpy as np
import math
import cv2
import mediapipe as mp


###############################################################################
class GazeEstimation():
    """This class has several methods to estimate gaze orientation, 
       plot face contours, face meshs, irises contours, pupils centres. 
    """

#------------------------------------------------------------------------------
    def __init__(self, input_image):
        """GazeEstimation class requires one input which is the input image
        """

        self.input_image = input_image

#------------------------------------------------------------------------------        
    def extract_face_landmarks(self):
        """This method returns arrays of face landmarks, irises landmarks as follow:
           landmarks, face_landmarks : represent all face landmarks where (face_landmarks = landmarks.landmark)
           irises_landmarks_pointer : represents the indexes of the irises landmarks inside the face_landmarks array
           left_eye_landmarks_pointer: represents the indexes of the lefet eye landmarks inside the face_landmarks array
           right_eye_landmarks_pointer: represents the indexes of the right eye landmarks inside the face_landmarks array
        """        
              
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5) as face_mesh:
        
            results = face_mesh.process(self.input_image)
            
            if results.multi_face_landmarks:
                irises_landmarks_pointer    = mp_face_mesh.FACEMESH_IRISES
                left_eye_landmarks_pointer  = mp_face_mesh.FACEMESH_LEFT_EYE   
                right_eye_landmarks_pointer = mp_face_mesh.FACEMESH_RIGHT_EYE  
                landmarks                   = results.multi_face_landmarks[0] 
                face_landmarks              = landmarks.landmark
                               
                
            else:
                print("No landmarks detected")
                landmarks                   = []
                face_landmarks              = []
                irises_landmarks_pointer    = [] 
                left_eye_landmarks_pointer  = [] 
                right_eye_landmarks_pointer = [] 
        return landmarks, face_landmarks, irises_landmarks_pointer, left_eye_landmarks_pointer, right_eye_landmarks_pointer

#------------------------------------------------------------------------------           
    def get_left_pupil_centre(self): 
        """This method returns the (x, y) point of the left eye pupil centre
        """ 
        
        _, face_landmarks, irises_landmarks_pointer, _, _ = self.extract_face_landmarks()
        image                                             = self.input_image.copy()
        
        if face_landmarks:

            irises_landmarks_pointer = list(irises_landmarks_pointer)   
            xt, yr                   = irises_landmarks_pointer[0] 
            xb, yl                   = irises_landmarks_pointer[1]       
            xl, yt                   = irises_landmarks_pointer[5] 
            xr, yb                   = irises_landmarks_pointer[6] 
            
            """      irises landmarks and pupil centre (x, y)
            
                           (xt, yt)
                          /        \
                        /           \
                (xl, yl)   (x, y)   (xr, yr)
                       \            /
                        \          /
                          (xb, yb)
            
 
            """

            # Denormalise the extracted Four irises landmarks based on the input image dimensions        
            yt = int(face_landmarks[yt].y * image.shape[0])
            xt = int(face_landmarks[xt].x * image.shape[1])
        
            yb = int(face_landmarks[yb].y * image.shape[0])
            xb = int(face_landmarks[xb].x * image.shape[1])
        
            yr = int(face_landmarks[yr].y * image.shape[0])
            xr = int(face_landmarks[xr].x * image.shape[1])
        
            yl = int(face_landmarks[yl].y * image.shape[0])
            xl = int(face_landmarks[xl].x * image.shape[1])
        
            # the pupil centre (x, y) is the mean of the four irises landmarks
            pupil_x = int(np.mean([xt, xr, xl, xb]))
            pupil_y = int(np.mean([yt, yr, yl, yb]))  
            
        else:
            print("No landmarks detected")
            pupil_x                = []
            pupil_y           = []
           
            
        return pupil_x, pupil_y      

#------------------------------------------------------------------------------
    def get_right_pupil_centre(self): 
        """This method returns the (x, y) point of the left eye pupil centre
        """ 
        
        _, face_landmarks, irises_landmarks_pointer, _, _ = self.extract_face_landmarks()
        image                                       = self.input_image.copy()
        
        if face_landmarks:
            irises_landmarks_pointer = list(irises_landmarks_pointer)   
            xt, yr                   = irises_landmarks_pointer[2] 
            xb, yl                   = irises_landmarks_pointer[3]       
            xl, yt                   = irises_landmarks_pointer[4] 
            xr, yb                   = irises_landmarks_pointer[7] 

            """      irises landmarks and pupil centre (x, y)
            
                           (xt, yt)
                          /        \
                        /           \
                (xl, yl)   (x, y)   (xr, yr)
                       \            /
                        \          /
                          (xb, yb)
            
 
            """        
            
            # Denormalise the extracted Four irises landmarks based on the input image dimensions       
            yt = int(face_landmarks[yt].y * image.shape[0])
            xt = int(face_landmarks[xt].x * image.shape[1])
        
            yb = int(face_landmarks[yb].y * image.shape[0])
            xb = int(face_landmarks[xb].x * image.shape[1])
        
            yr = int(face_landmarks[yr].y * image.shape[0])
            xr = int(face_landmarks[xr].x * image.shape[1])
        
            yl = int(face_landmarks[yl].y * image.shape[0])
            xl = int(face_landmarks[xl].x * image.shape[1])
        
            # the pupil centre (x, y) is the mean of the four irises landmarks
            pupil_x = int(np.mean([xt, xr, xl, xb]))
            pupil_y = int(np.mean([yt, yr, yl, yb])) 
            
        else:
            print("No landmarks detected")
            pupil_x                = []
            pupil_y           = []
            
        return pupil_x, pupil_y 

#------------------------------------------------------------------------------
    def plot_face_mesh(self):    
        """This method returns the input image with face mesh plotted over it
        """ 
        
        landmarks, _, _, _, _ = self.extract_face_landmarks()
        
        mp_drawing        = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh      = mp.solutions.face_mesh
        image             = self.input_image.copy()
        
        mp_drawing.draw_landmarks(
            image                   = image,
            landmark_list           = landmarks,
            connections             = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec   = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_tesselation_style())        
        return image
    
#------------------------------------------------------------------------------    
    def plot_face_contours(self):    
        """This method returns the input image with face contours plotted over it
        """ 
        
        landmarks, _, _, _, _ = self.extract_face_landmarks()
        
        mp_drawing        = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh      = mp.solutions.face_mesh
        image             = self.input_image.copy()
        
        mp_drawing.draw_landmarks(
            image                   = image,
            landmark_list           = landmarks,
            connections             = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec   = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_contours_style())
        return image

#------------------------------------------------------------------------------    
    def plot_irises_landmarks(self):    
        """This method returns the input image with irises contours plotted over it
        """ 
        
        landmarks, _, _, _, _ = self.extract_face_landmarks()
        
        mp_drawing        = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh      = mp.solutions.face_mesh
        image             = self.input_image.copy()
        
        mp_drawing.draw_landmarks(
            image                   = image,
            landmark_list           = landmarks,
            connections             = mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec   = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())        
        return image



#------------------------------------------------------------------------------    
    def plot_pupils_centres(self):    
        """This method returns the input image with pupils centres plotted over it
        """
        
        left_pupil_x,  left_pupil_y  = self.get_left_pupil_centre()
        right_pupil_x, right_pupil_y = self.get_right_pupil_centre()
        
        image = self.input_image.copy()
        
        if left_pupil_x and left_pupil_y and right_pupil_x and right_pupil_y:
            image = self.plot_eyes_contours()                       
            image = self.plot_plus(image, left_pupil_x,  left_pupil_y)
            image = self.plot_plus(image, right_pupil_x, right_pupil_y)       
        return image

#------------------------------------------------------------------------------    
    def plot_plus(self, image, x, y):       
        """This method returns the input image with (+) symbol plotted over the given (x, y) point
        """
        color         = (0, 255, 255) # yellow color
        linelength    = 3
        
        # start and end point of the horizontal line of + symbol
        start_point_h = (x - linelength, y) 
        end_point_h   = (x + linelength, y)
        
        # start and end point of the vertical line of + symbol
        start_point_v = (x, y - linelength) 
        end_point_v   = (x, y + linelength)
        
        cv2.line(image, start_point_h, end_point_h, color)        
        cv2.line(image, start_point_v, end_point_v, color)
        return image

#------------------------------------------------------------------------------    
    def write_pupils_centres(self):
        """This method returns a black image with centre points of left and right pupils written over it
        """

        left_pupil_x,  left_pupil_y  = self.get_left_pupil_centre()
        right_pupil_x, right_pupil_y = self.get_right_pupil_centre()
        
        image      = self.input_image.copy()
        text_image = np.zeros(image.shape)
        
        # paramters to set the font properties, colour, location of the pupil centres text 
        txt_location1 = (90, 130)
        txt_location2 = (90, 165)
        color         = (147, 58, 31)
        thickness     = 1
        font_scale    = 0.9
        font          = cv2.FONT_HERSHEY_DUPLEX
        
        if left_pupil_x and left_pupil_y and right_pupil_x and right_pupil_y:
            left_txt  = "Left pupil:   " + str((left_pupil_x, left_pupil_y))
            right_txt = "Right pupil: "  + str((right_pupil_x, right_pupil_y))
        else:
            left_txt  = "Left pupil:   ( , )" 
            right_txt = "Right pupil: ( , )"
                
        cv2.putText(text_image, left_txt, txt_location1, font, font_scale, color, thickness)
        cv2.putText(text_image, right_txt, txt_location2, font, font_scale, color, thickness)
        return text_image  

#------------------------------------------------------------------------------
    def plot_eyes_contours(self):    
        """This method returns the input image with both eyes contours plotted over it
        """
        
        landmarks, _, _, _, _ = self.extract_face_landmarks()
        
        mp_drawing            = mp.solutions.drawing_utils
        mp_drawing_styles     = mp.solutions.drawing_styles
        mp_face_mesh          = mp.solutions.face_mesh
        image                 = self.input_image.copy()
        
        mp_drawing.draw_landmarks(
            image                   = image,
            landmark_list           = landmarks,
            connections             = mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec   = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_contours_style())
        
        mp_drawing.draw_landmarks(
            image                   = image,
            landmark_list           = landmarks,
            connections             = mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec   = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_contours_style())
        return image         

#------------------------------------------------------------------------------
    def get_eyes_boundingbox(self):    
        """This method returns points of the bounding boxes for both eyes as follow:
        left_eye  bounding box    returned as [Ymin, Ymax, Xmin, Xmax]
        right_eye bounding box    returned as [Ymin, Ymax, Xmin, Xmax]
        """        
        _, face_landmarks, _,  left_eye_landmarks_pointer, right_eye_landmarks_pointer = self.extract_face_landmarks()
        image = self.input_image.copy()
        
        if face_landmarks:
            """the index numbers 
            (386, 374, 362, 263)(Ymin, Ymax, Xmin, Xmax) and 
            (159, 145, 33, 133)(Ymin, Ymax, Xmin, Xmax)
            are the top-left and bottom-right corners of the left eye and 
            right eye coordinates respectively            
            """
            # Extract the bounding boxes pints and Denormalise them on the input image dimensions    
            left_eye_miny = int(face_landmarks[386].y * image.shape[0])
            left_eye_maxy = int(face_landmarks[374].y * image.shape[0])
            
            left_eye_minx = int(face_landmarks[362].x * image.shape[1])            
            left_eye_maxx = int(face_landmarks[263].x * image.shape[1])
            
            right_eye_miny = int(face_landmarks[159].y * image.shape[0])
            right_eye_maxy = int(face_landmarks[145].y * image.shape[0])
            
            right_eye_minx = int(face_landmarks[33].x * image.shape[1])            
            right_eye_maxx = int(face_landmarks[133].x * image.shape[1])
          
        else:
            left_eye_miny = []
            left_eye_maxy = []
            left_eye_minx = []
            left_eye_maxx = []
            
            right_eye_miny = []
            right_eye_maxy = []
            right_eye_minx = []
            right_eye_maxx = []
           
        left_eye  = [left_eye_miny, left_eye_maxy, left_eye_minx, left_eye_maxx]
        right_eye = [right_eye_miny, right_eye_maxy, right_eye_minx, right_eye_maxx]
        return left_eye, right_eye 

#------------------------------------------------------------------------------
    def horizontal_vertical_blinking_gaze_ratios(self):
            """Returns horizontal eyes ratio, vertical eyes ratio, left and right eyes blink ratios
            Each ratio is a number between 0.0 and 1.0 that indicates the
            horizontal, vertical direction of the gaze and the blinking ratio of each eye as follow:
            
            horizontal ratio: extreme right direction (= 0.0), centre (= 0.5), extreme left direction (= 1.0)
            vertical ratio: extreme top direction (= 0.0), centre (= 0.5), extreme bottom direction (= 1.0)
            left/right eyes blink ratios: blinknig (<= 0.3), open (> 0.3)
            """
            
            left_eye, right_eye          = self.get_eyes_boundingbox()
            left_pupil_x,  left_pupil_y  = self.get_left_pupil_centre()
            right_pupil_x, right_pupil_y = self.get_right_pupil_centre()
            
            if left_pupil_x and  left_pupil_y and right_pupil_x and right_pupil_y:
                h_left_range  = np.abs(left_eye[3] - left_eye[2])
                h_right_range = np.abs(right_eye[3] - right_eye[2])
                
                h_pupil_left  = np.abs(left_pupil_x  - left_eye[2])  / h_left_range
                h_pupil_right = np.abs(right_pupil_x - right_eye[2]) / h_right_range
                hori_ratio    = np.mean([h_pupil_left, h_pupil_right])
                
                #..............................
                
                v_left_range  = np.abs(left_eye[1] - left_eye[0])
                v_right_range = np.abs(right_eye[1] - right_eye[0])            
                           
                v_pupil_left  = np.abs(left_pupil_y  - left_eye[0])  / v_left_range
                v_pupil_right = np.abs(right_pupil_y - right_eye[0]) / v_right_range
                vert_ratio    = np.mean([v_pupil_left, v_pupil_right])
                
                #..............................
                
                maxi = 21 # maximum eye vertical range when eye is fully open
                
                blink_ratio_left_eye  = v_left_range / maxi
                blink_ratio_right_eye = v_right_range / maxi
            else:
                hori_ratio            = []
                vert_ratio            = []
                blink_ratio_left_eye  = []
                blink_ratio_right_eye = []
            
            return hori_ratio, vert_ratio, blink_ratio_left_eye, blink_ratio_right_eye

#------------------------------------------------------------------------------ 
    def estimate_gaze_direction(self):
        """This method returns a black image with text indicating gaze direction written over it
        """
        
        hori_ratio, vert_ratio, blink_ratio_left_eye, blink_ratio_right_eye = self.horizontal_vertical_blinking_gaze_ratios()
        
        blink_condition = 0.3          # left/right eyes blink ratios: blinknig (<= 0.3), open (> 0.3)
        hori_gaze_range = [0.45, 0.55] # right direction (<= 0.45), left direction (>= 0.55), centre (else)
        vert_gaze_range = [0.45, 0.55] # top direction (<= 0.45), bottom direction (>= 0.55), centre (else)
        
        
        """ prepare indexes to pick the text that descibes the gaxe orientation from the gaze array below. 
        This is based on the computed horizontal, vertical, blinking ratios
        """
        # Initilise indexes
        blinking_condition  = -1
        hori_gaze_condition = -1
        vert_gaze_condition = -1
        
        #  Set indexes based on the computed horizontal, vertical, blinking ratios
        if blink_ratio_left_eye and blink_ratio_right_eye:
            if       blink_ratio_left_eye <= blink_condition and not blink_ratio_right_eye <= blink_condition: 
                blinking_condition = 0
            
            elif not blink_ratio_left_eye <= blink_condition and     blink_ratio_right_eye <= blink_condition:
                blinking_condition = 2
            
            elif     blink_ratio_left_eye <= blink_condition and     blink_ratio_right_eye <= blink_condition:
                blinking_condition = 1  
           
            else:   
                if hori_ratio:                   
                    if   hori_ratio <= hori_gaze_range[0]:
                        hori_gaze_condition = 2
                    
                    elif hori_ratio >= hori_gaze_range[1]:
                        hori_gaze_condition = 0
                    
                    else:
                        hori_gaze_condition = 1
                if vert_ratio:            
                    if   vert_ratio <= vert_gaze_range[0]:
                        vert_gaze_condition = 0
                  
                    elif vert_ratio >= vert_gaze_range[1]:
                        vert_gaze_condition = 2
                   
                    else:
                        vert_gaze_condition = 1
      

             
        #  Gaze direction array
        gaze_direction = (["Looking top left",     "Looking top",             "Looking top right"],
                          ["Looking left",         "Looking centre",          "Looking right"],
                          ["Looking bottom left",  "Looking bottom",          "Looking bottom right"],
                          ["Left eye is blinking", "Boths eyes are blinking", "Right eye is blinking"])
            
        """
         0,0 | 0,1 | 0,2
        -----------------
         1,0 | 1,1 | 1,2
        -----------------
         2,0 | 2,1 | 2,2
        -----------------
         3,0 | 3,1 | 3,2
        
        """
        
        # pick the best text that descibes the gaze orientation using the prepared indexes
        if blinking_condition > -1:
            text = gaze_direction[3][blinking_condition]     
        else:
            if hori_gaze_condition> -1 and vert_gaze_condition > -1:
                text = gaze_direction[vert_gaze_condition][hori_gaze_condition]             
            else:
                text = "..."

        
        
        # paramters to set the font properties, colour, location
        image        = self.input_image.copy()
        text_image   = np.zeros(image.shape)
        txt_location = (90, 60)        
        color        = (147, 58, 31)
        thickness    = 2
        font_scale   = 1.6
        font         = cv2.FONT_HERSHEY_DUPLEX        
        
        cv2.putText(text_image, text, txt_location, font, font_scale, color, thickness)
        return text_image
    
#------------------------------------------------------------------------------
    def plot_gaze_direction(self):
        """This method returns a black image with text indicating gaze direction written over it.
           It also returns a visualisation of the gaze direction of both eyes as two cirles 
           with arrows in side them pointong to the gize direction, or fully coloured to indicate eyes blinking.
        """
        
        hori_ratio, vert_ratio, blink_ratio_left_eye, blink_ratio_right_eye = self.horizontal_vertical_blinking_gaze_ratios()
        
        blink_condition = 0.3          # left/right eyes blink ratios: blinknig (<= 0.3), open (> 0.3)
        hori_gaze_range = [0.45, 0.55] # right direction (<= 0.45), left direction (>= 0.55), centre (else)
        vert_gaze_range = [0.45, 0.55] # top direction (<= 0.45), bottom direction (>= 0.55), centre (else)        
        

        """ prepare indexes to pick the text that descibes the gaxe orientation from the gaze array below. 
        This is based on the computed horizontal, vertical, blinking ratios
        """
        # Initilise indexes to control the text and gaze visualisation 
        blinking_condition  = -1
        hori_gaze_condition = -1
        vert_gaze_condition = -1
        
        Lthickness = 1
        Rthickness = 1
        angle = '0'
        
        show_blinking_left  = [-1, -1, 1]
        show_blinking_right = [1, -1, -1]
        
        
        
        
        
        #  Set indexes based on the computed horizontal, vertical, blinking ratios
        if blink_ratio_left_eye and blink_ratio_right_eye:
            if       blink_ratio_left_eye <= blink_condition and not blink_ratio_right_eye <= blink_condition: 
                blinking_condition = 0
            
            elif not blink_ratio_left_eye <= blink_condition and     blink_ratio_right_eye <= blink_condition:
                blinking_condition = 2
            
            elif     blink_ratio_left_eye <= blink_condition and     blink_ratio_right_eye <= blink_condition:
                blinking_condition = 1  
           
            else:   
                if hori_ratio:                   
                    if   hori_ratio <= hori_gaze_range[0]:
                        hori_gaze_condition = 2
                    
                    elif hori_ratio >= hori_gaze_range[1]:
                        hori_gaze_condition = 0
                    
                    else:
                        hori_gaze_condition = 1
                if vert_ratio:            
                    if   vert_ratio <= vert_gaze_range[0]:
                        vert_gaze_condition = 0
                  
                    elif vert_ratio >= vert_gaze_range[1]:
                        vert_gaze_condition = 2
                   
                    else:
                        vert_gaze_condition = 1
      

             
        #  Gaze direction array
        gaze_direction = (["Looking top left",     "Looking top",             "Looking top right"],
                          ["Looking left",         "Looking centre",          "Looking right"],
                          ["Looking bottom left",  "Looking bottom",          "Looking bottom right"],
                          ["Left eye is blinking", "Boths eyes are blinking", "Right eye is blinking"])

        # the gaze angle array is used in ploting the arrows that visualise the gaze direction
        gaze_angle    = ([90 + 45,    90,        45],
                         [180,       '0',         0],
                         [180 + 45,  270,  270 + 45],
                         ['0',       '0',       '0'])
        
        
        
        """
         0,0 | 0,1 | 0,2
        -----------------
         1,0 | 1,1 | 1,2
        -----------------
         2,0 | 2,1 | 2,2
        -----------------
         3,0 | 3,1 | 3,2
        
        """
   
        # pick the best text that descibes the gaze orientation using the prepared indexes
        if blinking_condition > -1:
            text = gaze_direction[3][blinking_condition]    
            Lthickness = show_blinking_left[blinking_condition]
            Rthickness = show_blinking_right[blinking_condition]
        else:
            if hori_gaze_condition> -1 and vert_gaze_condition > -1:
                text  = gaze_direction[vert_gaze_condition][hori_gaze_condition]
                angle = gaze_angle[vert_gaze_condition][hori_gaze_condition]
            else:
                text = "..."

        
        
        # paramters to set the font properties, colour, location
        image        = self.input_image.copy()
        text_image   = np.zeros(image.shape)
        txt_location = (90, 60)        
        color        = (147, 58, 31)
        thickness    = 2
        font_scale   = 1.6
        font         = cv2.FONT_HERSHEY_DUPLEX        
        
        cv2.putText(text_image, text, txt_location, font, font_scale, color, thickness)
        
        #--------------------------------------------------------------------------------
       
        # paramters to set colour, location of the eyes visualisation cirles and gaze arrows
        Leftcenter  = [100, 250]
        Rightcenter = [250, 250]
        radius      = 50
        color1      = (255, 0, 0)
        thickness   = 2
        
        Lstart_point = Leftcenter.copy()        
        Lend_point = Lstart_point.copy()
        
        Rstart_point = Rightcenter.copy()        
        Rend_point = Rstart_point.copy()
        
        
        if type(angle) != str:
            Lend_point =  np.add(Lend_point, self.pol2cart(radius, -angle))
            Rend_point =  np.add(Rend_point, self.pol2cart(radius, -angle))
          
       
        text_image = cv2.arrowedLine(text_image, Lstart_point, Lend_point, color1, thickness) 
        text_image = cv2.arrowedLine(text_image, Rstart_point, Rend_point, color1, thickness) 
          
        cv2.circle(text_image, Leftcenter, radius, color1, Lthickness, lineType=8, shift=0)
        cv2.circle(text_image, Rightcenter, radius, color1, Rthickness, lineType=8, shift=0)
 
        #-------------------------------
        left_pupil_x,  left_pupil_y  = self.get_left_pupil_centre()
        right_pupil_x, right_pupil_y = self.get_right_pupil_centre()
        
        # paramters to set the font properties, colour, location of the pupil centres text       
        txt_location1 = (90, 130)
        txt_location2 = (90, 165)
        color         = (147, 58, 31)
        thickness     = 1
        font_scale    = 0.9
        font          = cv2.FONT_HERSHEY_DUPLEX
        
        if left_pupil_x and left_pupil_y and right_pupil_x and right_pupil_y:
            left_txt  = "Left pupil:   " + str((left_pupil_x, left_pupil_y))
            right_txt = "Right pupil: " + str((right_pupil_x, right_pupil_y))
        else:
            left_txt  = "Left pupil:   ( , )" 
            right_txt = "Right pupil: ( , )"
                
        cv2.putText(text_image, left_txt, txt_location1, font, font_scale, color, thickness)
        cv2.putText(text_image, right_txt, txt_location2, font, font_scale, color, thickness)
        
        return text_image
           
           
#------------------------------------------------------------------------------
    def pol2cart(self, radius, angle):
        """ This method convert polar cordinates (radius, angle) to cartesian (x, y)
        """
        x = int(radius * math.cos(math.radians(angle)))
        y = int(radius * math.sin(math.radians(angle)))
        return [x, y]        
        

               



