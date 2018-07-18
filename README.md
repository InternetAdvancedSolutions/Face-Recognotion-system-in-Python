# Face_Recognition-App-
Recognises the given sample image from given list of person(Open CV)


* This application recognises the image of any person from the given imagesof all.
* Here the data_folder_has has to included before execution of the script.
* This folder has should contain the folders caontaing the various images of all the person.
* For good training , the image folder should contain large set of images.
* Here using Open CV in python environment as working tool.
* Supervised machine learning model is uesd for predicting , to which label the image falls to.
* For recognition and training purpose , machine learning concepts has too been used.
* Instead fo haar cascasde, we are using lbp_cascade , although it's a bit slow but gives the better result.


_____________________________________________________________________________________________________

*Various steps taken are:

1) Training our model, from the images of various given set.
2) Detection of the face from the sample image.
3) extracting the various features from the sample detected image.
4) predicting the image to get the label from our data_folder i.e name of person.

______________________________________________________________________________________________________

* More the number of images better will be the training, hence better results..................
