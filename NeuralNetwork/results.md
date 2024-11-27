### PyTorch Neural Network
Specific changes made from the previous testing results:
- removed some columns, such as 'description' that are not possible to OneHotEncode. Before the presentation I will try to perform some NLP techniques on it so they can be included in the calculations
- added a column (actors_count), that grabs the number of actors that are in the movie and also present in a top 100 actors dataset.
- implemented a randomized search cross validation technique to determine the chosen best parameters that I then trained the entire model on


Results:
- The neural network results are much worse than that of other models, it decreasing loss over epochs, but not enough to really dent the huge error it's providing.
- The mean absolute area is in the 200  millions, and the R2 score is less than -5

Learning Graph: learning.png