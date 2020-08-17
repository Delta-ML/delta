/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package utils

import (
	"delta/delta-serving/core/types"
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
