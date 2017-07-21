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

type Perceptron struct {
	learningRate float64
	activationFunc ActivationFunc
	weights Weights
	totalErrors int
}

//NewPerceptron creates new perceptron instance 
func NewPerceptron(learningRate float64, activationFunc ActivationFunc, weights Weights,) Perceptron {
	return Perceptron{
		learningRate: learningRate,
		activationFunc: activationFunc,
		weights: weights,
	}
}

func (p *Perceptron) Process( learningDataSet DataSet, expectedResult ResultSet, iterations int) (amountOfErrors, iterationsDone int) {
	for ; iterationsDone < iterations; iterationsDone++ {
		iterErrors := p.ProcessOnce(learningDataSet, expectedResult)
		if iterErrors == 0 {
			return amountOfErrors, iterationsDone
		}

		amountOfErrors += iterErrors
	}

	return amountOfErrors, iterationsDone
}

func (p *Perceptron) ProcessOnce(dataSet DataSet, expectedResult ResultSet) (amountOfErrors int) {

	currentErrors := p.totalErrors;
	for index, dataLine := range dataSet {
		p.ProcessOneDataLine(dataLine, expectedResult[index])
	}

	return p.totalErrors - currentErrors
}

func (p *Perceptron) ProcessOneDataLine(dataLine DataLine, expectedResult float64) {

	resultError := expectedResult - p.Evaluate(dataLine)
	updatedWeights := make(Weights, len(p.weights))

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

func (p Perceptron) Evaluate(dataLine DataLine) float64 { 
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