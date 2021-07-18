# Neural-Network-Student-Grades

## Description 
This project features a Neural Network with 7 layers trained on student grade data. Which 
predicts student grades out of 20 within 1.01 to 1.3 points off. 

## Installation
* Pip install h5py (built with 3.1.0)
* Pip install tensorflow (built with 2.5.0)
* Pip install numpy (built with 1.19.5)
* Download below file, and place downloaded file in same directory as the python files,  
  rename the downloaded file to "student_grades.csv"  
  https://www.kaggle.com/dipam7/student-grade-prediction


## Usage
Before running any python files please make sure you have completed ALL above installation steps. 

The file testing_student_grades.py features a neural network trained on student data acquired from
Kaggle. The structure of the model is contained in this file as well as data engineering and 
cleaning. 

The file testing_my_grades.py features neural networks trained on my first year grades, 
used to predict my next year grade(s). The purpose of this, was to review my own knowledge 
so far.

Folder, 'images', contains different results of training the model with different epoch values on the 
student data. 

## Neural Network Details
Layer (Type) | Output Shape | Params  
dense (Dense)                (None, 256)               8448      
leaky_re_lu (LeakyReLU)      (None, 256)               0         
dense_1 (Dense)              (None, 192)               49344     
leaky_re_lu_1 (LeakyReLU)    (None, 192)               0         
dense_2 (Dense)              (None, 128)               24704     
leaky_re_lu_2 (LeakyReLU)    (None, 128)               0         
dense_3 (Dense)              (None, 64)                8256      
leaky_re_lu_3 (LeakyReLU)    (None, 64)                0         
dense_4 (Dense)              (None, 32)                2080      
leaky_re_lu_4 (LeakyReLU)    (None, 32)                0         
dense_5 (Dense)              (None, 1)                 33        
leaky_re_lu_5 (LeakyReLU)    (None, 1)                 0         
dense_6 (Dense)              (None, 1)                 2         
  
RMSE Testing Error: 1.01 - 1.3 / 20 (See images/r9.PNG)

## Credits
* Author: Lee Taylor
