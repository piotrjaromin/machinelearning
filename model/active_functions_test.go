package model

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestSpec(t *testing.T) {


	Convey("HeavySideFunc should return correct values", t, func() {

		So(HeavySideStepFunc(-0.5), ShouldEqual, -1)
		So(HeavySideStepFunc(-10), ShouldEqual, -1)
		So(HeavySideStepFunc(-4), ShouldEqual, -1)
		So(HeavySideStepFunc(0), ShouldEqual, 1)
		So(HeavySideStepFunc(1), ShouldEqual, 1)
		So(HeavySideStepFunc(.01), ShouldEqual, 1)
	})

}