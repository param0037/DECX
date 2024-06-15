; ---------------------------------------------------------------------
; Author : Wayne Anderson
; Date   : 2021.04.16
; ---------------------------------------------------------------------
; This is a part of the open source program named "DECX", copyright c Wayne,
; 2021.04.16, all right reserved.
; More information please visit https://github.com/param0037/DECX


; eax = 16H -> eax returns base freq (/MHz)
;              ebx returns max freq (/MHz)
;              ecx returns bus freq (/MHz)
; eax = 0BH -> ebx returns # of logical processors (with ecx = 01H)
; Cache size: obtained by multiplying all the (figures + 1) together in EBX and ECX.

.CODE
CPUID_call PROC
    
    xor     eax,        eax
    xor     ebx,        ebx
    xor     edx,        edx

    mov     r8,         rcx
    lea     r9,         [r8]
    mov     eax,        DWORD PTR   [r9]
    mov     ecx,        DWORD PTR   [r9 + 8]

    cpuid

    mov     [r9],       DWORD PTR   eax
    mov     [r9 + 4],   DWORD PTR   ebx
    mov     [r9 + 8],   DWORD PTR   ecx
    mov     [r9 + 12],  DWORD PTR   edx
    ret

CPUID_call ENDP

END