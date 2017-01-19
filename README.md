# Vehicle-identification
Use CNN to identify vehicles from surveillance video.

This is the final project for course `Principles of Machine Learning`(机器学习原理). The task is to identify a car in the surveillance video: not cars of the same model, but exactly the same car.


The network structure is as following:

![network-structure](https://github.com/soloice/Vehicle-identification/raw/master/network-structure.png)

To be precise, we use triplet loss for this task. For any input "anchor"(an image of a car), pick a positive example(another image of the same car) and a negative example(an image of another car), and input these 3 images to the same CNN to get their "deepID". Then we require the distance between the anchor and positive example to be less than that between the anchor and negative example. The margin bewteen these 2 distances are a linear combination of several parameters, i.e. the if "negative" and "anchor" are of the same color, if they are of the same model. Except for the main loss, we also used an auxilary softmax output which trys to predict the ID of input image itself. These 2 losses are combined linearly. In summary, we use softmax classifer to let the CNN to recognize a car, and use triplet loss to let it distinguish different cars. These 2 different losses share the same basic part, thus improves the generalization ability of our model.

Finally, as a retrieval problem, we generate the deepID for all images in the database. For any given image, we return images which have the closest deepIDs as suspected candidates of the given one in the database.
