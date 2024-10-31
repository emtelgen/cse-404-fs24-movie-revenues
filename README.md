# How to use
- Run model.py

# Team Members
- Emily Telgenhoff
- Dung (Zoom) Nguyen
- Advaith Vuppala
- Seth Darling

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
> Please make sure to give us access to your code using our GitHub ids (Kordjamshidi,Sanya1001,afsharim).
