package neurons

import (
	"log"
	m "github.com/piotrjaromin/machine-learning/model"
)

type Neuron interface {
	Process( learningDataSet m.DataSet, expectedResult m.ResultSet) (amountOfErrors int)
	Evaluate(dataLine m.DataLine) float64
}
type Adaline struct {
	learningRate float64
	activationFunc m.ActivationFunc
	weights m.Weights
	totalErrors int
}

//NewAdaline creates new Adaline instance 
func NewAdaline(learningRate float64, activationFunc m.ActivationFunc, weights m.Weights,) Adaline {
	return Adaline{
		learningRate: learningRate,
		activationFunc: activationFunc,
		weights: weights,
	}
}

func (p *Adaline) Process(dataSet m.DataSet, expectedResult m.ResultSet) (amountOfErrors int) {

	currentErrors := 0
	costFuncWeights := make(m.Weights, len(p.weights))
	for index, dataLine := range dataSet {
		expectedVal := expectedResult[index]
		
		result := p.weights[0]
		for index, dataVar := range dataLine {
			result += dataVar*p.weights[index + 1]
		}

		diffResult := expectedVal - result
		costFuncWeights[0] += diffResult
		for index, dataVal := range dataLine {
			costFuncWeights[index+1] += diffResult*dataVal
		}

		if diffResult != 0.0 {
			currentErrors++
		}
	}

	//Adaline updates weights after processing all dataś
	for index := range p.weights {
		// w_i = w_i + delta_w
		p.weights[index] += p.learningRate*costFuncWeights[index]
	}

	p.totalErrors += currentErrors
	return currentErrors
}

func (p Adaline) Evaluate(dataLine m.DataLine) float64 { 
	//weight with zero index is for x_0=1
	if len(p.weights) != len(dataLine) + 1 {
		log.Panic("weights should len(weights) == len(dataLine) + 1")
	}

	result := p.weights[0]
	for index, dataVar := range dataLine {
			result += dataVar*p.weights[index + 1]
	}

	return p.activationFunc(result)
}

func (p Adaline) TotalErrors() int {
	return p.totalErrors
}
