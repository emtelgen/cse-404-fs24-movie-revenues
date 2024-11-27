### Gradient Boosting Model (model1.png)
Specific changes made from the previous testing results:
- removed some columns, such as 'description' that are not possible to OneHotEncode. Before the presentation I will try to perform some NLP techniques on it so they can be included in the calculations
- added a column (actors_count), that grabs the number of actors that are in the movie and also present in a top 100 actors dataset.
- implemented a randomized search cross validation technique to determine the chosen best parameters that I then trained the entire model on
Results:
- Test Mean Absolute Error: 954395 (better than the previous one which was over 20 million)
- R squared score hovering around .9976.. ie all the features I have chosen are directly impacting the results

Learning Graph: learning.png