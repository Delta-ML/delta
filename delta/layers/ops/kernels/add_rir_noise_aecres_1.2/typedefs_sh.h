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

#ifndef __TYPEDEFS_SH_H_
#define __TYPEDEFS_SH_H_

const double c = 340.0f;  //	sound speed

typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long DWORD;
typedef int BOOL;
typedef char CHAR;
typedef short SHORT;
typedef long LONG;
typedef unsigned long ULONG;
typedef LONG HRESULT;

#define _MAX_PATH 260 /*  max. length of full pathname */

#define S_OK ((HRESULT)0L)
#define S_FALSE ((HRESULT)1L)

#define FALSE false
#define TRUE true

#ifndef PI
#define PI 3.1415926535f
#endif

#define DECLARE_HANDLE(name) \
  struct name##__ {          \
    int unused;              \
  };                         \
  typedef struct name##__ *name

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef EPSILON
#define EPSILON 1e-5
#endif

#define RIR_LENGTH 16000

typedef struct {
  float real;
  float image;
} COMPLEX;

#endif  //__TYPEDEFS_SH_H_
