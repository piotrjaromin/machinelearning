package neurons

import (
	. "github.com/smartystreets/goconvey/convey"
	"log"
	"testing"
	"os"
	"encoding/csv"
	"bufio"
	"io"
	"strconv"
	m "github.com/piotrjaromin/machine-learning/model"
)

func TestSpec(t *testing.T) {

	const weightDivider = 10
	const amountOfIter = 10
	weights := m.InitWeights(5, weightDivider)

	Convey("Perceptron should correctly learn to recognize setosa and versicolor", t, func() {
		const learningRate = 0.25
		wholeDataSet, expResults := readData()
		p := NewPerceptron(learningRate, m.HeavySideStepFunc, weights)
		testNeuron(&p, wholeDataSet, expResults, amountOfIter)
	})

	Convey("Adaline should correctly learn to recognize setosa and versicolor", t, func() {
		const learningRate = 0.0001
		wholeDataSet, expResults := readData()
		p := NewAdaline(learningRate, m.HeavySideStepFunc, weights)
		testNeuron(&p, wholeDataSet, expResults, amountOfIter)
	})

}

func testNeuron(n Neuron,  wholeDataSet m.DataSet, expResults []float64, amountOfIter int) {

		learningDataSet := wholeDataSet[:40]
		learningDataSet = append(learningDataSet, wholeDataSet[50:90]...)

		learnedResults := expResults[:40]
		learnedResults = append(learnedResults, expResults[50:90]...)

		notLearnedDataSet := wholeDataSet[40:49]
		notLearnedDataSet = append(notLearnedDataSet, wholeDataSet[90:]...)

		notLearnedResults := expResults[40:49]
		notLearnedResults = append(notLearnedResults, expResults[90:]...)

		errors, iterationsDone := Teach(n, learningDataSet, learnedResults, amountOfIter)
		
		log.Printf("errors during processing %d\n", errors)
		log.Printf("iterations done %d\n", iterationsDone)
		//Check if it can predict correctly on unseen data
		for dataIndex := 0; dataIndex < len(notLearnedResults); dataIndex++ {
			evalResult := n.Evaluate(notLearnedDataSet[dataIndex])
			So(evalResult, ShouldEqual, notLearnedResults[dataIndex])
		}
}

func readData() (dataSet m.DataSet, expectedResults []float64) {

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

		dataLine := m.DataLine{}
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
