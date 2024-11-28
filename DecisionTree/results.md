### Decision Tree Regression Model
Specific changes made from the previous testing results:
- Implemented a Grid Search cross validation technique to determine the best chosen parameters that the model was then trained with. Some of the hyper-parameters included maximum depth of the regression tree, the minimum samples we split each region on, the minimum samples a "leaf" region must have, maximum features, and the ccp_alpha used for post-pruning the tree.
- did not include actor for this specific run through as the process crashed my computer. 
Results:
- Test Mean Absolute Error: 2190585.35 This is an improvement from 2270267.34
- R squared for test was: 0.985117870031461, which is an improvement from 0.9692087802075433

Learning Graph: learning.png