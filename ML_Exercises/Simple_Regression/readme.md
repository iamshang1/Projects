#Polyfit Regression

This exercise attempts to extrapolate website views using different degrees of polynomial regression
on previous web traffic data.

#### Extrapolating Future Website Views

![extrapolation](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Simple_Regression/extrapolation.png)

#### Adjusted R2 on Training Data for Different Degrees of Polynomial Regression

![adjustedr2](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Simple_Regression/adjustedr2.png)

#### Polyfit Accuracy in Unshuffled K-Folds Validation with 15 Folds

![crossvalidation](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Simple_Regression/crossvalidation.png)

**Note:** the average R2 is negative because nearly all of the polynomial fits perform poorly on folds containing the tail end of the data.

Based on the graphs above, it seems that using 4 or 5 degree polynomial regression provides the best extrapolation.