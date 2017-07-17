package perceptron

import (
	"log"
	"math/rand"
	"time"
)

type Weights []float64
type DataLine []float64
type DataSet []DataLine
type ResultSet []float64

type ActivationFunc func(float64)float64

type perceptron struct {
	learningRate float64
	activationFunc ActivationFunc
}

func NewPerceptron(learningRate float64, activationFunc ActivationFunc) perceptron {
	return perceptron{
		learningRate: learningRate,
		activationFunc: activationFunc,
	}
}

func (p perceptron) Process(weights Weights, learningDataSet DataSet, expectedResult ResultSet, iterations int) Weights{
		for i:=0; i < iterations; i++ {
			weights = p.ProcessOnce(weights, learningDataSet, expectedResult)
		}

		return weights
}

func (p *perceptron) ProcessOnce(weights Weights, dataSet DataSet, expectedResult ResultSet) Weights {

	for index, dataLine := range dataSet {
		weights = p.ProcessOneDataLine(weights, dataLine, expectedResult[index])
	}

	return weights
}

func (p perceptron) ProcessOneDataLine(weights Weights, dataLine DataLine, expectedResult float64) Weights {

	result := p.Evaluate(weights, dataLine)
	resultError := expectedResult - result
	updatedWeights := make(Weights, len(weights))

	for index, weight := range weights {
		// delta_w = learningRate * (y - y') * x_i
		delta := p.learningRate*(resultError)*dataLine[index]

		// w_i = w_i + delta_w
		updatedWeights[index] = weight + delta
	}

	return updatedWeights
}

func (p perceptron) Evaluate(weights Weights, dataLine DataLine) float64 { 
	if len(weights) != len(dataLine) {
		log.Panic("weights should have the same length as dataLine")
	}

	result := 0.0
	for index, dataVar := range dataLine {
		result += dataVar*weights[index]
	}

	return p.activationFunc(result)
}


func HeavySideStepFunc(x float64) float64 {
	if x >=0 {
		return 1
	}

	return -1
}

func InitWeights(amount int, divider float64) (w Weights) {
    s := rand.NewSource(time.Now().UnixNano())
    r := rand.New(s)

	for index := 0; index < amount; index++ {
		 w = append(w, r.Float64() / divider )
	}
	return 
}