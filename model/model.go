package model

type Weights []float64
type DataLine []float64
type DataSet []DataLine
type ResultSet []float64

type ActivationFunc func(float64)float64