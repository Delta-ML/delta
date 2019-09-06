package handel

import (
	. "delta/deltann/server/model"
	"github.com/gin-gonic/gin"
	"net/http"
)

// Binding from JSON
type DeltaRequest struct {
	DeltaRawText       string        `form:"delta_raw_text" json:"delta_raw_text"`
	DeltaSignatureName string        `form:"signature_name" json:"signature_name" `
	DeltaInstances     []interface{} `form:"instances" json:"instances" `
	DeltaInputs        interface{}   `form:"inputs" json:"inputs" `
}

func DeltaPredictHandler(context *gin.Context) {
	var json DeltaRequest
	if err := context.ShouldBindJSON(&json); err != nil {
		context.JSON(http.StatusBadRequest, gin.H{"error": "DeltaRequest information is not complete"})
		return
	}

	modelResult, err := DeltaModelRun(json.DeltaRawText)
	if err != nil {
		context.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	context.JSON(http.StatusOK, gin.H{"predictions": modelResult})
}
