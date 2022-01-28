# Apply-Lipstick-based-on-Facial-Landmark-Detection
Apply arbitrary lipstick colors to a face image based on facial landmark detection

## Algorithm Outline:
* Detect the face region and facial landmarks.  
* Extract landmarks around the lip region and generate a mask for the lip region.  
* Change the color of the lip region, applying Gaussian blurring and seamless cloning to preserve the texture of the lip.

## Video Demo:  
https://youtu.be/XOA7W6yfNDE

## Image Demo 1: Before (left) and After (right) Applying Lipstick  
![alt text](https://github.com/yyhz76/Apply-Lipstick-based-on-Facial-Landmark-Detection/blob/main/demo/demo1.png)<br /><br />  

### Facial Landmarks (left) and Lip Mask (right)
![alt text](https://github.com/yyhz76/Apply-Lipstick-based-on-Facial-Landmark-Detection/blob/main/demo/landmark_and_lip_mask.png)<br /><br /> 

### Gaussian Blur Off (left) vs Gaussian Blur On (right)
Blurring the lip boundary region make the lip appear more natural after applying the color change
![alt text](https://github.com/yyhz76/Apply-Lipstick-based-on-Facial-Landmark-Detection/blob/main/demo/no_blur_vs_has_blur.png)<br /><br />  


### Seamless Cloning Off (left) vs Seamless Cloning On (right)
Seamless Cloning preserves the texture of the lip after applying the color change, making the image more realistic
![alt text](https://github.com/yyhz76/Apply-Lipstick-based-on-Facial-Landmark-Detection/blob/main/demo/no_seamlessCloning_vs_has_seamlessCloning.png)<br /><br />  

## Image Demo 2: Before (left) and After (right) Applying Lipstick     
![alt text](https://github.com/yyhz76/Apply-Lipstick-based-on-Facial-Landmark-Detection/blob/main/demo/demo2.png)<br /><br />  


Image copyrights belong to https://learnopencv.com/
