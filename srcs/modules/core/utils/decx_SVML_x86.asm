;   ----------------------------------------------------------------------------------
;   Author : Wayne Anderson
;   Date   : 2021.04.16
;   ----------------------------------------------------------------------------------
; 
; This is a part of the open source project named "DECX", a high-performance scientific
; computational library. This project follows the MIT License. For more information 
; please visit https://github.com/param0037/DECX.
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


.DATA
    Abs_int_sign    dd      7fffffffH
    ONE_over_PI     dd      0.3183098862f
    Minus_Pi        dd      -3.1415926536f
    ONE_fp32        dd      1.f
    TWO_fp32        dd      2.f
    Quarter_pi      dd      0.7853981634f

    Halv_fp32       dd      00800000H
    ONE_int32       dd      01H
    
    ONE_over_2      dd      0.5f

    COS_TAYLOR      dd      0.0833333333f
                    dd      0.0333333333f
                    dd      0.0178571429f
                    dd      0.0111111111f
                    dd      0.0075757576f
                    dd      0.0059523810f
                    dd      0.0041666667f
                    dd      0.0032679739f
                    dd      0.0026315789f
                    dd      0.0021645022f
                    dd      0.0018115942f



; __vectorcall convention
.CODE
; COS_TAYLOR_CORE PROC


; COS_TAYLOR_CORE ENDP

__cos_fp32x4@@16 PROC
    
    push            rbx
    movups          DWORD PTR   [rsp - 16],  XMM6
    movups          DWORD PTR   [rsp - 32],  XMM7
    movups          DWORD PTR   [rsp - 48],  XMM8

    mov             rax,    7fffffff7fffffffH               
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1                    ; XMM1 -> inverted mask of a integer
    pand            XMM0,   XMM1                            ; x = abs(x) (XMM0 -> abs(x))

    mov             rax,    4513444118665558403               
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1
    vmulps          XMM2,   XMM0,   XMM1                    ; XMM2 -> x / pi
    vroundps        XMM2,   XMM2,   01H                     ; XMM2 -> floor(x/pi)           (preserved)

    mov             rax,    13855623165080309723               
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1                    ; XMM1 -> -pi
    vfmadd132ps     XMM1,   XMM0,   XMM2                    ; XMM1 -> x - period * pi (normalized angle) (preserved)

    ; Angles normalized

    mov             rax,    4596222329050697691
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                    ; XMM3 -> pi / 2
    
    mov             rax,    7fffffff7fffffffH               
    vmovq           XMM4,   rax
    vpunpcklqdq     XMM4,   XMM4,   XMM4                    ; XMM4 -> inverted mask of a integer
    vsubps          XMM3,   XMM1,   XMM3                    ; XMM3 -> norm(angle) - pi/2
    vandps          XMM3,   XMM4,   XMM3                    ; XMM3 -> abs(norm(angle) - pi/2)

    mov             rax,    4560193532023345115
    vmovq           XMM4,   rax
    vpunpcklqdq     XMM4,   XMM4,   XMM4                    ; XMM4 -> pi / 4
    vcmpps          XMM4,   XMM3,   XMM4,   01H             ; XMM4 -> sin_rectf             (preserved)

    mov             rax,    4618102649103240164
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                    ; XMM3 -> pi*3/4
    vcmpps          XMM5,   XMM1,   XMM3,   0EH             ; XMM5 -> cos_otherside         (preserved)
    
    mov             rax,    4596222329050697691
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                    ; XMM3 -> pi / 2
    vandps          XMM3,   XMM3,   XMM4                    ; XMM3 -> sin_rectf ? pi / 2 : 0
    vsubps          XMM1,   XMM1,   XMM3

    mov             rax,    4632251126078050267
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                    ; XMM3 -> pi
    vandps          XMM3,   XMM3,   XMM5
    vsubps          XMM1,   XMM1,   XMM3

    mov             rax,    8000000080000000H
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                    ; XMM3 -> 0x80000000
    vandps          XMM3,   XMM3,   XMM5
    vxorps          XMM1,   XMM1,   XMM3
    ; Pre-process of angle is finished

    mov             rax,    4575657222473777152
    vmovq           XMM0,   rax
    vpunpcklqdq     XMM0,   XMM0,   XMM0                    ; XMM0 -> 1.f

    vmulps          XMM3,   XMM1,   XMM1                    ; XMM3 -> x_sq                  (preserved)
    ; XMM0, XMM1, XMM2, XMM3, XMM4, and XMM5 are preserved

    mov             rax,    4539628425446424576
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6                    ; XMM6 -> 0.5f
    mov             rax,    4479580431832492715
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7                    ; XMM7 -> 0.1666666667f

    vblendvps       XMM6,   XMM6,   XMM7,   XMM4            ; XMM6 -> fact
    vmulps          XMM6,   XMM3,   XMM6                    ; XMM6 -> x_term                (preserved)
    vsubps          XMM0,   XMM0,   XMM6                    ; XMM0 -> res                   (preserved)

    mov             rax,    4443551634805140139
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7                    ; XMM7 -> 0.0833333333f
    mov             rax,    4417130516412419277
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8                    ; XMM8 -> 0.05f
    vblendvps       XMM7,   XMM7,   XMM8,   XMM4            ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7                    ; Update x_term
    vaddps          XMM0,   XMM0,   XMM6                    ; Update res

    mov             rax,    4397915159143155849
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7                    ; XMM7 -> 0.0333333333f
    mov             rax,    4378356668346600497
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8                    ; XMM8 -> 0.0238095238f
    vblendvps       XMM7,   XMM7,   XMM8,   XMM4            ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7                    ; Update x_term
    vsubps          XMM0,   XMM0,   XMM6                    ; Update res

    mov             rax,    4364631413154269477
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7                    ; XMM7 -> 0.0178571429f
    mov             rax,    4351478041447468601
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8                    ; XMM8 -> 0.0138888889f
    vblendvps       XMM7,   XMM7,   XMM8,   XMM4            ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7                    ; Update x_term
    vaddps          XMM0,   XMM0,   XMM6                    ; Update res

    mov             rax,    4338667803267959649
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7                    ; XMM7 -> 0.0111111111f
    mov             rax,    4329351263286522377
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8                    ; XMM8 -> 0.0090909091f
    vblendvps       XMM7,   XMM7,   XMM8,   XMM4            ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7                    ; Update x_term
    vsubps          XMM0,   XMM0,   XMM6                    ; Update res

    ; XMM3, XMM6, and XMM7 are free
    mov             rax,    8000000080000000H
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                    ; XMM3 -> 0x80000000
    vxorps          XMM1,   XMM1,   XMM3                    ; -norm(angle)

    mov             rax,    4575657222473777152
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6                    ; XMM6 -> 1.f
    vblendvps       XMM6,   XMM6,   XMM1,   XMM4
    vmulps          XMM0,   XMM0,   XMM6                    ; modified res by multiplying x when in sine case

    vbroadcastss    XMM1,   DWORD PTR   [ONE_int32]         ; XMM1 = <int>[1 1 1 1]
    cvtps2dq        XMM2,   XMM2                            ; XMM2 -> int(floor(period))
    pand            XMM2,   XMM1                            ; XMM2 = sign of full_period_num
    pslld           XMM2,   1FH                             ; XMM2 = mask of sign inversion
    mov             rax,    8000000080000000H
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6                    ; XMM6 -> 0x80000000
    vpand           XMM6,   XMM6,   XMM5
    vpxor           XMM2,   XMM2,   XMM6
    vpxor           XMM0,   XMM0,   XMM2                    ; XMM0 -> masked inversed

    movups          XMM6,   DWORD PTR   [rsp - 16]
    movups          XMM7,   DWORD PTR   [rsp - 32]
    movups          XMM8,   DWORD PTR   [rsp - 48]
    pop             rbx

    ret
    
__cos_fp32x4@@16 ENDP

PUBLIC __cos_fp32x4@@16


; __vectorcall convention
.CODE
__sin_fp32x4@@16 PROC
    
    push            rbx
    movups          DWORD PTR   [rsp - 16],  XMM10
    sub             rsp,    16

    mov             rax,    0                               ; 0
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1                    ; XMM1 -> pi / 2
    vcmpps          XMM10,  XMM0,   XMM1,   0EH             ; XMM10 -> angle > 0 ? 1<32> : 0<32>

    mov             rax,    4596222329050697691             ; pi / 2
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1                    ; XMM1 -> pi / 2
    vsubps          XMM0,   XMM1,   XMM0                    ; XMM0 = pi/2 - angle
    call            __cos_fp32x4@@16

    vandps          XMM0,   XMM0,   XMM10                   ; res = angle == 0 ? 0 : res

    movups          XMM10,  DWORD PTR   [rsp - 16]
    add             rsp,    16
    pop             rbx

    ret

__sin_fp32x4@@16 ENDP

PUBLIC __sin_fp32x4@@16


END