### Random Forest Regressor Model (Random_Forest_Model_Learning_Curve_Evaluation.jpg)
Specific changes made from the previous testing results:
- removed some columns, such as 'description' that are not possible to OneHotEncode. Before the presentation, I will try to perform some NLP techniques on it so they can be included in the calculations
- added a column (actors_count), that grabs the number of actors that are in the movie and also present in a top 100 actors dataset.
- implemented a randomized search cross validation technique to determine the chosen best parameters that I then trained the entire model on
Results:
- Test Mean Squared Error: 832020029058306.8
- Test Mean Absolute Error: 16506928.826625574

Learning Graph: Random_Forest_Model_Learning_Curve_Evaluation.jpg