package model

import (
	"math/rand"
	"time"
)

func InitWeights(amount int, divider float64) (w Weights) {
    s := rand.NewSource(time.Now().UnixNano())
    r := rand.New(s)

	for index := 0; index < amount; index++ {
		 w = append(w, r.Float64() / divider )
	}
	return 
}