# Eyes  Tracking Opencv python :eye: :eye:

Estimate the position of eyes in the frame using computer Vision Techniques.

## TODO 

- [ ] Detect Eyes LandMarks using Dlib Library(python)
- [ ] find the landmarks of eyes 
- [ ] Add blinking eyes detection 
- [ ] Extract Eyes using Masking Techniques 
- [ ] Apply Thresholding find White and Black pixels 
- [ ] find out the ratio of white and black pixels and estimate the position on the bases of ratio.
- [ ] improve the visual appearance of information 

## Installation 
Steps are involved to run the code.

### Requirements are :
1. install Dlib
    #### For Windows
    - Inorder to install dilb on windows machines you need following: :smirk:
        - Visual Studio
        - Visual Studio Build Tools
        - Cmake

    `pip install dlib`
    #### Linux or Mac OS
    - Just you need Cmake that all here on linux and Mac OS
    - install Dilb uisng Pip command
        `pip3 install dlib`
        
2. install opencv-python
    `pip install opencv-python`
    for linux or Mac OS replace`pip` with  `pip3` and you are good to go... :wink:

3. Download landmarks [Predictor](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)  
    - Extract that file place this into the pr
