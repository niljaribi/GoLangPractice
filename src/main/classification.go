package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
	"github.com/sjwhitworth/golearn/trees"
	"github.com/sjwhitworth/golearn/linear_models"
	"github.com/sjwhitworth/golearn/ensemble"
)

func main() {
	// Load in a dataset, with headers. Header attributes will be stored.
	// Think of instances as a Data Frame structure in R or Pandas.
	// You can also create instances from scratch.
	rawData, err := base.ParseCSVToInstances("C:/GoCode/testProject/src/main/GoldData.csv", false)
	if err != nil {
		panic(err)
	}
	// Print a pleasant summary of your data.
	fmt.Println(rawData)

	//Initialises a new KNN classifier
	clsTree := trees.NewRandomTree(10)
	clsKNN := knn.NewKnnClassifier("euclidean", "linear", 2)
	lr := linear_models.NewLinearRegression()
	clsRF := ensemble.NewRandomForest(10,118)
	clsKD := knn.NewKnnClassifier("euclidean", "kdtree", 2)

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	clsTree.Fit(trainData)
	clsKNN.Fit(trainData)
	lr.Fit(trainData)
	clsRF.Fit(trainData)
	clsKD.Fit(trainData)

	//Calculates the Euclidean distance and returns the most popular label
	predictionsTree, errTree := clsTree.Predict(testData)
	predictionsKNN, errKNN := clsKNN.Predict(testData)
	predictionLR, errLR := lr.Predict(testData)
	predictionRF, errRF := clsRF.Predict(testData)
	predictionsKD, errKD := clsKD.Predict(testData)

	if errTree != nil {
		panic(errTree)
	}
	if errKNN != nil {
		panic(errKNN)
	}
	if errLR != nil {
		panic(errLR)
	}
	if errRF != nil {
		panic(errRF)
	}
	if errKD != nil {
		panic(errTree)
	}
	// Prints precision/recall metrics
	confusionMatTree, errTree := evaluation.GetConfusionMatrix(testData, predictionsTree)
	confusionMatKNN, errKNN := evaluation.GetConfusionMatrix(testData, predictionsKNN)
	confusionMatLR, errLR := evaluation.GetConfusionMatrix(testData,predictionLR)
	confusionMatRF, errRF := evaluation.GetConfusionMatrix(testData,predictionRF)
	confusionMatKD, errKD := evaluation.GetConfusionMatrix(testData,predictionsKD)
	if errTree != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", errTree.Error()))
	}
	if errKNN != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", errKNN.Error()))
	}
	if errLR != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", errLR.Error()))
	}
	if errRF != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", errLR.Error()))
	}
	//if errSVC != nil {
	//	panic(fmt.Sprintf("Unable to get confusion matrix: %s", errSVC.Error()))
	//}
	fmt.Println("Tree classifier")
	fmt.Println(evaluation.GetSummary(confusionMatTree))
	fmt.Println("KNN classifier")
	fmt.Println(evaluation.GetSummary(confusionMatKNN))
	fmt.Println("LR classifier")
	fmt.Println(evaluation.GetSummary(confusionMatLR))
	fmt.Println("RF classifier")
	fmt.Println(evaluation.GetSummary(confusionMatRF))
	fmt.Println("KD classifier")
	fmt.Println(evaluation.GetSummary(confusionMatKD))
}