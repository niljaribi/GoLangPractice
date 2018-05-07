package main 

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/linear_models"
)
func main() {
	lr := NewLinearRegression()
	rawData, err := base.ParseCSVToInstances("C:/GoCode/testProject/src/main/GoldData.csv", true)
	//So(err, ShouldBeNil)
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	lr.Fit(trainData)
	predictionLR, err := lr.Predict(testData)
	confusionMatLR, errLR := evaluation.GetConfusionMatrix(testData,predictionLR)
	fmt.Println(evaluation.GetSummary(confusionMatLR))
	
}
