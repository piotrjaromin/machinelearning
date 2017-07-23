package neurons

import (
	m "github.com/piotrjaromin/machine-learning/model"
)

func Teach(n Neuron, learningDataSet m.DataSet, expectedResult m.ResultSet, iterations int) (amountOfErrors, iterationsDone int) {
	for ; iterationsDone < iterations; iterationsDone++ {
		iterErrors := n.Process(learningDataSet, expectedResult)
		if iterErrors == 0 {
			return amountOfErrors, iterationsDone
		}

		amountOfErrors += iterErrors
	}

	return amountOfErrors, iterationsDone
}