; ----------------------------------------------------------------------------------
; Author : Wayne Anderson
; Date : 2021.04.16
; ----------------------------------------------------------------------------------
;
; This is a part of the open source project named "DECX", a high-performance scientific
; computational library. This project follows the MIT License. For more information
; please visit https:
;
; Copyright (c) 2021 Wayne Anderson
;
; Permission is hereby granted, free of charge, to any person obtaining a copy of this
; software and associated documentation files (the "Software"), to deal in the Software
; without restriction, including without limitation the rights to use, copy, modify,
; merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
; permit persons to whom the Software is furnished to do so, subject to the following
; conditions:
;
; The above copyright notice and this permission notice shall be included in all copies
; or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
; INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
; PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
; FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
; OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
; DEALINGS IN THE SOFTWARE.
; eax = 16H -> eax returns base freq (/MHz)
; ebx returns max freq (/MHz)
; ecx returns bus freq (/MHz)
; eax = 0BH -> ebx returns # of logical processors (with ecx = 01H)
; Cache size: obtained by multiplying all the (figures + 1) together in EBX and ECX.
; __stdcall ---- called funciton should save E(R)BX, E(R)SI, E(R)DI, E(R)BP registers, if used
default rel
CPUID_call:
    xor eax, eax
    push rbx
    xor ebx, ebx
    xor edx, edx
    mov r8, rdi
    lea r9, [r8]
    mov eax, DWORD [r9]
    mov ecx, DWORD [r9 + 8]
    cpuid
    mov [r9], DWORD eax
    mov [r9 + 4], DWORD ebx
    mov [r9 + 8], DWORD ecx
    mov [r9 + 12], DWORD edx
    pop rbx
    ret
GLOBAL CPUID_call
