**Exercise Questions in Lab PCA**

*   **Exercise:** What is stored in `.coef_` and `.intercept_`? Why are there so many of them?
*   **Exercise:** Hmm, cross-validation didn't seem to offer improved results. Is this correct? Is it possible for cross-validation to not yield better results than non-cross-validation? If so, how and why?
*   **Exercise:** Why didn't we scale the y-values (class labels) or transform them with PCA? Is this a mistake?
*   **Exercise:** Our data only has 2 dimensions/features now. What do these features represent?
*   **Exercise:** Critique the PCA plot. Does it prove that wines are similar? Why/why not?
*   **Exercise:** The wine data we've used so far consist entirely of continuous predictors. Would PCA work with categorical data? Why or why not?
*   **Exercise:** Clusters overlap despite PCA. What could cause this [two disjoint clusters in the PCA plot when colored by quality]? What does this mean? 
*   **Exercise:** Wow. Look at that separation [by wine color]. Too bad we aren't trying to predict if a wine is red or white. Does this graph help you answer our previous question [about the disjoint clusters]? Does it change your thoughts? What new insights do you gain?
*   **Exercise:** Use Logistic Regression (with and without cross-validation) on the PCA-transformed data. Do you expect this to outperform our original 75% accuracy? What are your results? Does this seem reasonable?
*   **Exercise:**
    1.  Fit a PCA that finds the first 10 PCA components of our training data
    2.  Use `np.cumsum()` to print out the variance we'd be able to explain by using n PCA dimensions for n=1 through 10
    3.  Does the 10-dimension PCA agree with the 2d PCA on how much variance the first components explain? **Do the 10d and 2d PCAs find the same first two dimensions? Why or why not?**
    4.  Make a plot of number of PCA dimensions against total variance explained. What PCA dimension looks good to you?
*   **Exercise:** Looking at your graph [of cumulative variance explained], what is the 'elbow' point / how many PCA components do you think we should use? Does this number of components imply that predictive performance will be optimal at this point? Why or why not?