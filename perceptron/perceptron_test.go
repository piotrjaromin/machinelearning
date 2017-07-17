package perceptron

import (
	. "github.com/smartystreets/goconvey/convey"
	"log"
	"testing"
	"os"
	"encoding/csv"
	"bufio"
	"io"
	"strconv"
)

func TestSpec(t *testing.T) {

	const learningRate = 0.01
	const amountOfIter = 10

	Convey("Perceptron should correctly learn to recognize setosa and versicolor", t, func() {

		wholeDataSet, expResults := readData()
		learningDataSet := wholeDataSet[:40]
		learningDataSet = append(learningDataSet, wholeDataSet[50:90]...)

		learnedResults := expResults[:40]
		learnedResults = append(learnedResults, expResults[50:90]...)

		notLearnedDataSet := wholeDataSet[40:49]
		notLearnedDataSet = append(notLearnedDataSet, wholeDataSet[90:]...)

		notLearnedResults := expResults[40:49]
		notLearnedResults = append(notLearnedResults, expResults[90:]...)
		weights := InitWeights(4, 10)

		p := NewPerceptron(learningRate, HeavySideStepFunc)

		weights = p.Process(weights, learningDataSet, learnedResults, amountOfIter)
		
		//Check if it can predict correctly on unseen data
		for dataIndex := 0; dataIndex < len(notLearnedResults); dataIndex++ {
			evalResult := p.Evaluate(weights, notLearnedDataSet[dataIndex])
			So(evalResult, ShouldEqual, notLearnedResults[dataIndex])
		}
	})

	Convey("HeavySideFunc should return correct values", t, func() {

		So(HeavySideStepFunc(-0.5), ShouldEqual, -1)
		So(HeavySideStepFunc(-10), ShouldEqual, -1)
		So(HeavySideStepFunc(-4), ShouldEqual, -1)
		So(HeavySideStepFunc(0), ShouldEqual, 1)
		So(HeavySideStepFunc(1), ShouldEqual, 1)
		So(HeavySideStepFunc(.01), ShouldEqual, 1)
	})

}

func readData() (dataSet DataSet, expectedResults []float64) {

	file, err := os.Open("./setosa_versicolor.data")
	if err != nil {
		log.Panic("Could not load test data file", err)
	}

	data := bufio.NewReader(file)
	r := csv.NewReader(data)

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Panic("Coult not read data line", err)
		}

		dataLine := DataLine{}
		recordLen := len(record)
		for index := 0; index < recordLen - 1; index++ {
			val, err := strconv.ParseFloat(record[index], 64)
			if err != nil {
				log.Panic("Could not parse line ", record, ". Details: ", err)
			}
			dataLine = append(dataLine, val)
		}

		dataSet = append(dataSet, dataLine)
		if record[recordLen - 1] == "Iris-setosa" {
			expectedResults = append(expectedResults, 1)
		} else {
			expectedResults = append(expectedResults, -1)
		}
	}
	return 
}
