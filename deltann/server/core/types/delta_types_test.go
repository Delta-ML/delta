package types

import (
	"gopkg.in/go-playground/assert.v1"
	"testing"
)

func TestDeltaTypes(t *testing.T) {
	dev := Develop
	assert.Equal(t, dev, "develop")
}
