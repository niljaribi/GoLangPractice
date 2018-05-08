data:

The data is obtained from investing.com
Memory should be involved somewhere in training the model. In this experiment, 14 previous days are considered in creating the feature space.
The data is normalized and features are extracted. 

The feature space will be Returns of today with response to j days in the past and mean returns of today till j days in the past.

Defining Returni:

Returni = (Pricei - Price(i-j))/Price(i-j)
MeanReturns = (Returni + Return(i-1)+...+ Return(i-j))/j

The amount of change was calculated for all the columns (Open, High, and Low)

Classification:

Different models were utilized for predicting the change in the gold price. The best result is achieved by Linear regression using both Scikit and GoLang.

Performance:
The performance of the regression model using GoLang is 99.85% and using Scikit is 83.28%.
