package neurons

import (
	"log"
	m "github.com/piotrjaromin/machine-learning/model"
)

type Perceptron struct {
	learningRate float64
	activationFunc m.ActivationFunc
	weights m.Weights
	totalErrors int
}

//NewPerceptron creates new perceptron instance 
func NewPerceptron(learningRate float64, activationFunc m.ActivationFunc, weights m.Weights,) Perceptron {
	return Perceptron{
		learningRate: learningRate,
		activationFunc: activationFunc,
		weights: weights,
	}
}


func (p *Perceptron) Process(dataSet m.DataSet, expectedResult m.ResultSet) (amountOfErrors int) {

	currentErrors := p.totalErrors;
	for index, dataLine := range dataSet {
		p.ProcessOneDataLine(dataLine, expectedResult[index])
	}

	return p.totalErrors - currentErrors
}

func (p *Perceptron) ProcessOneDataLine(dataLine m.DataLine, expectedResult float64) {

	resultError := expectedResult - p.Evaluate(dataLine)
	updatedWeights := make(m.Weights, len(p.weights))

	//because x_0 = 1
	updatedWeights[0] = p.learningRate*resultError
	for index, weight := range p.weights[1:] {
		// delta_w = learningRate * (y - y') * x_i
		delta := p.learningRate*(resultError)*dataLine[index]

		// w_i = w_i + delta_w
		updatedWeights[index + 1] = weight + delta
	}

	if (resultError != 0) {
		p.totalErrors++
	} 

	p.weights = updatedWeights
}

func (p Perceptron) Evaluate(dataLine m.DataLine) float64 { 
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

func (p Perceptron) TotalErrors() int {
	return p.totalErrors
}
