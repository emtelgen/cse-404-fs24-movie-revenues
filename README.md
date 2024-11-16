# How to use
- Run model.py

# Team Members
- Emily Telgenhoff
- Dung (Zoom) Nguyen
- Advaith Vuppala
- Seth Darling
- Ephraim Bennett

# Project 1 Step 4: Data Reader & Starting Initial Model

- [x] Step 1 is the reader and you need to have that correctly done. 
- [x] Step 2 is more for your exploration and learning at this stage, try to do something about it as much as you can, a naive initial model would be ok here.

## Step 1: 
- Data Reader is a program that reads data items from files and folders into your machine learning program, and creates training/testing examples. Depending on how clean/structured is your data this step might need a bit of work for you.
> Please, push the code to your GitHub repository and post the commit's URL here in D2L. 

## Step 2: 
- Initial Model. When you have the data reader in place, use a learning algorithm on the data, and evaluate its performance, even a naive initial working model that is made by your team would be sufficient.  Also, please write up the following information: 

- What is the setting of your project? is it supervised, unsupervised, or semisupervised? 
- - Our project is supervised with our model being trained on labled movie data.
- How you represent X (Features/Representation  and their explanation, justification)
 - - X is our set of input features that includes various attributes of movies such as the average rating, duration, director, release data, genres, language and additional columns from the actors, to crew data. These features are represented numerically, either directly (duration) or converted (genres/languages), using techniques like on-hot encoding for categrial features.
- How do you represent Y (binary? multi-class? sequences? trees? graphs?)
 - - Y is our target variable, revenue, which is represented as a continuous numerical value measured in US dollars. The output is modeled as a regression problem since revenue is a continuous variable.
- Which libraries you will use?
 - - We are using Sklearn for the model and Pandas for handling the data.

> Note that, all this information will be a part of your final project report.

### You might use any machine learning or deep learning libraries for your project or even use your own implemented algorithm, in any case, you will need to do the following:

#### Make a test/train split of the data
2. Use your machine learning algorithm to learn from your training set
3. Evaluate the model with the test set.
4. Send a report summary about the model's performance on train and test datasets (it is fine that it is very low at this stage).
5. Push your code to your repository on GitHub, and post a link to it here in response to this assignment.
> Please make sure to give us access to your code using our GitHub ids (Kordjamshidi,Sanya1001,afsharim,Sethdarling1s).
# Project Step 5
## Decision Tree
A decision Tree was used to predict the movie revenue. This was done using sci-kit learn's class for the model. There is performance data for both the training set and the testing set. For the training set, the Mean Absolute Error was 2,146,931.4910654337 and the R^2 was 0.9716725076412114. This indicates our model fits the data very well, and is only off by a few million which is small relative to the expected values. For the testing set it was was slightly less accurate but still sufficient. Mean Absolute Error was 2270267.3394089667 and R^2 was 0.9692087802075433 which is only marginally less accurate. Nearly all variance is captured by this decision tree.


### Gradient Boosting Model (model1.png)
Another model chosen to use was the Gradient Boosting Model. This model was tried and hyperparemeters chosen to reduce the Mean Absolute Error and increase the R^2 value, to 28166880 for the mean absolute error, and an R^2 value of .83. The hyperparemeters were adjusted based on what worked best with the data set, and as the model1.png shows, the line closely follows the values of the movie revenues. Most of the variance was captured by this model as well.

### Random Forest Regressor Model 
The Random Forest Regressor model from sklearn was used on training and testing datasets. The model produced an R^2 value of 0.9326532462988735 and had Mean Absolute Error of 12012789.964526067. The model does a relatively good job of predicting the box office revenue. The hyperparameters were adjusted to reduce the variance and to control the tree's growth. For the testing data set, the model had a Mean Absolute Error of 14236243.853435155 and R^2 of 0.912465378346. The graph illustrates how the accurate the evaluation of the model is with the predicted revenue vs the actual revenue along with high evaluation metrics.

### SGD Model
We chose to use a Stochastic Gradient Descent model to see how it would compare to original linear regression model. This model did not perform well with a Mean Absolute Error of 65580930.60790263 and a R^2 of 0.054907837369382095. Capturing some varience but not much.

### LightGBM model
The results from the LightGBM model shows that the model generally underestimates the revenue. There are some outliers such as the actual revenue of 8 that is far from the predicted revenue. The LightGBM model seems to be better at getting the general trend in the data. Though, compared to the Random Forest model here many of the points lie exactly on the line of prediction, the accuracy of LightGBM may not be as good.