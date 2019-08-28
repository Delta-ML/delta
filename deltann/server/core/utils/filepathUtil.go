package utils

import (
	"delta/deltann/server/core/types"
	"path"
	"path/filepath"
	"runtime"
)

func GetCurrentPath() string {
	_, filename, _, _ := runtime.Caller(1)
	return path.Dir(filename)
}

func GetProjectPath(profile string) string {
	if profile != types.Develop {
		filePath, _ := filepath.Abs(`.`)
		return filePath
	}
	currentPath := GetCurrentPath()
	return path.Join(currentPath, `../../`)
}
