;
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






.DATA



    Abs_int_sign_dw dd      7fffffffH
    Mask_MSB_dw     dd      80000000H
    ONE_int32       dd      01H
    ONE_fp32        dd      1065353216   ; 1.f
    Pi              dd      1078530011   ; 3.1415926536f
    ONE_over_PI     dd      1050868099   ; 0.3183098862f
    Minus_Pi        dd      3226013659   ; -3.1415926536f
    Halv_Pi         dd      1070141403   ; 1.5707963268f
    Quarter_pi      dd      1061752795   ; 0.7853981634f
    Three_4_Pi      dd      1075235812   ; 2.3561944902f

    COS_TAYLOR      dd      1056964608   ; 0.5f
                    dd      1034594987   ; 0.0833333333f
                    dd      1023969417   ; 0.0333333333f
                    dd      1016219941   ; 0.0178571429f
                    dd      1010174817   ; 0.0111111111f

    SIN_TAYLOR      dd      1042983595   ; 0.1666666667f
                    dd      1028443341   ; 0.05f
                    dd      1019415601   ; 0.0238095238f
                    dd      1013157433   ; 0.0138888889f
                    dd      1008005641   ; 0.0090909091f





.CODE





















































































_avx_cos_fp32x8@@32 PROC
    vmovups  DWORD PTR [rsp - 32],     YMM6
    vmovups  DWORD PTR [rsp - 64],     YMM7
    vmovups  DWORD PTR [rsp - 96],     YMM8
    vbroadcastss    YMM1, DWORD PTR    [Abs_int_sign_dw] ; YMM1 -> inverted mask of a integer
    vpand           YMM0,   YMM0,   YMM1 ; x = abs(x) (YMM0 -> abs(x))
    vbroadcastss    YMM1, DWORD PTR    [ONE_over_PI]
    vmulps          YMM2,   YMM0,   YMM1 ; YMM2 -> x / pi
    vroundps        YMM2,   YMM2,   01H ; YMM2 -> floor(x/pi) (preserved)
    vbroadcastss    YMM1, DWORD PTR    [Minus_Pi] ; YMM1 -> -pi
    vfmadd132ps     YMM1,   YMM0,   YMM2 ; YMM1 -> x - period * pi (normalized angle) (preserved)
    ; Angles normalized
    vbroadcastss    YMM3, DWORD PTR    [Halv_Pi] ; YMM3 -> pi / 2
    vbroadcastss    YMM4, DWORD PTR    [Abs_int_sign_dw] ; YMM4 -> 7fffffffH
    vsubps          YMM3,   YMM1,   YMM3 ; YMM3 -> norm(angle) - pi/2
    vandps          YMM3,   YMM4,   YMM3 ; YMM3 -> abs(norm(angle) - pi/2)
    vbroadcastss    YMM4, DWORD PTR    [Quarter_pi] ; YMM4 -> pi / 4
    vcmpps          YMM4,   YMM3,   YMM4, 01H ; YMM4 -> sin_rectf (preserved)
    vbroadcastss    YMM3, DWORD PTR    [Three_4_Pi] ; YMM3 -> pi*3/4
    vcmpps          YMM5,   YMM1,   YMM3, 0EH ; YMM5 -> cos_otherside (preserved)
    vbroadcastss    YMM3, DWORD PTR    [Halv_Pi] ; YMM3 -> Pi / 2
    vandps          YMM3,   YMM3,   YMM4 ; YMM3 -> sin_rectf ? pi / 2 : 0
    vsubps          YMM1,   YMM1,   YMM3
    vbroadcastss    YMM3, DWORD PTR    [Pi] ; YMM3 -> pi
    vandps          YMM3,   YMM3,   YMM5
    vsubps          YMM1,   YMM1,   YMM3
    vbroadcastss    YMM3, DWORD PTR    [Mask_MSB_dw] ; YMM3 -> 0x80000000
    vandps          YMM3,   YMM3,   YMM5
    vxorps          YMM1,   YMM1,   YMM3
    ; Pre-process of angle is finished
    vbroadcastss    YMM0, DWORD PTR    [ONE_fp32] ; YMM0 -> 1.f
    vmulps          YMM3,   YMM1,   YMM1 ; YMM3 -> x_sq (preserved)
    ; YMM0, YMM1, YMM2, YMM3, YMM4, and YMM5 are preserved
    vbroadcastss    YMM6, DWORD PTR    [COS_TAYLOR] ; YMM6 -> 0.5f
    vbroadcastss    YMM7, DWORD PTR    [SIN_TAYLOR] ; YMM7 -> 0.1666666667f
    vblendvps       YMM6,   YMM6,   YMM7,   YMM4 ; YMM6 -> fact
    vmulps          YMM6,   YMM3,   YMM6 ; YMM6 -> x_term (preserved)
    vsubps          YMM0,   YMM0,   YMM6 ; YMM0 -> res (preserved)
    vbroadcastss    YMM7, DWORD PTR    [COS_TAYLOR + 4] ; YMM7 -> 0.0833333333f
    vbroadcastss    YMM8, DWORD PTR    [SIN_TAYLOR + 4] ; YMM8 -> 0.05f
    vblendvps       YMM7, YMM7, YMM8, YMM4 ; YMM7 -> fact
    vmulps          YMM7,   YMM7,   YMM3
    vmulps          YMM6,   YMM6,   YMM7 ; Update x_term
    vaddps          YMM0,   YMM0,   YMM6 ; Update res
    vbroadcastss    YMM7, DWORD PTR    [COS_TAYLOR + 8] ; YMM7 -> 0.0333333333f
    vbroadcastss    YMM8, DWORD PTR    [SIN_TAYLOR + 8] ; YMM8 -> 0.0238095238f
    vblendvps       YMM7, YMM7, YMM8, YMM4 ; YMM7 -> fact
    vmulps          YMM7,   YMM7,   YMM3
    vmulps          YMM6,   YMM6,   YMM7 ; Update x_term
    vsubps          YMM0,   YMM0,   YMM6 ; Update res
    vbroadcastss    YMM7, DWORD PTR    [COS_TAYLOR + 12] ; YMM7 -> 0.0178571429f
    vbroadcastss    YMM8, DWORD PTR    [SIN_TAYLOR + 12] ; YMM8 -> 0.0138888889f
    vblendvps       YMM7,   YMM7,   YMM8, YMM4 ; YMM7 -> fact
    vmulps          YMM7,   YMM7,   YMM3
    vmulps          YMM6,   YMM6,   YMM7 ; Update x_term
    vaddps          YMM0,   YMM0,   YMM6 ; Update res
    vbroadcastss    YMM7, DWORD PTR    [COS_TAYLOR + 16] ; YMM7 -> 0.0111111111f
    vbroadcastss    YMM8, DWORD PTR    [SIN_TAYLOR + 16] ; YMM8 -> 0.0090909091f
    vblendvps       YMM7,   YMM7,   YMM8, YMM4 ; YMM7 -> fact
    vmulps          YMM7,   YMM7,   YMM3
    vmulps          YMM6,   YMM6,   YMM7 ; Update x_term
    vsubps          YMM0,   YMM0,   YMM6 ; Update res
    ; YMM3, YMM6, and YMM7 are free
    vbroadcastss    YMM3, DWORD PTR    [Mask_MSB_dw] ; YMM3 -> 0x80000000
    vxorps YMM1,    YMM1,   YMM3 ; -norm(angle)
    vbroadcastss    YMM6, DWORD PTR    [ONE_fp32] ; YMM6 -> 1.f
    vblendvps YMM6, YMM6,   YMM1,   YMM4
    vmulps          YMM0,   YMM0,   YMM6 ; modified res by multiplying x when in sine case
    vbroadcastss YMM1, DWORD PTR       [ONE_int32] ; YMM1 = <int>[1 1 1 1]
    vcvtps2dq       YMM2,   YMM2 ; YMM2 -> int(floor(period))
    vpand           YMM2,   YMM2,   YMM1 ; YMM2 = sign of full_period_num
    vpslld YMM2,    YMM2, 1FH ; YMM2 = mask of sign inversion
    vbroadcastss    YMM6, DWORD PTR    [Mask_MSB_dw] ; YMM6 -> 0x80000000
    vpand           YMM6,   YMM6,   YMM5
    vpxor           YMM2,   YMM2,   YMM6
    vpxor           YMM0,   YMM0,   YMM2 ; YMM0 -> masked inversed
    vmovups         YMM6, DWORD PTR [rsp - 32]
    vmovups         YMM7, DWORD PTR [rsp - 64]
    vmovups         YMM8, DWORD PTR [rsp - 96]
    ret

_avx_cos_fp32x8@@32 ENDP










_avx_cos_fp32x4@@16 PROC
    push rbx
    movups   DWORD PTR [rsp - 16],     XMM6
    movups   DWORD PTR [rsp - 32],     XMM7
    movups   DWORD PTR [rsp - 48],     XMM8
    mov             rax,    7fffffff7fffffffH
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1                ; XMM1 -> inverted mask of a integer
    pand            XMM0,   XMM1                        ; x = abs(x) (XMM0 -> abs(x))
    mov             rax,    4513444118665558403
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1
    vmulps          XMM2,   XMM0,   XMM1                ; XMM2 -> x / pi
    vroundps        XMM2,   XMM2,   01H                 ; XMM2 -> floor(x/pi) (preserved)
    mov             rax,    13855623165080309723
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1                ; XMM1 -> -pi
    vfmadd132ps     XMM1,   XMM0,   XMM2                ; XMM1 -> x - period * pi (normalized angle) (preserved)
    ; Angles normalized
    mov             rax,    4596222329050697691
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                ; XMM3 -> pi / 2
    mov             rax,    7fffffff7fffffffH
    vmovq           XMM4,   rax
    vpunpcklqdq     XMM4,   XMM4,   XMM4                ; XMM4 -> inverted mask of a integer
    vsubps          XMM3,   XMM1,   XMM3                ; XMM3 -> norm(angle) - pi/2
    vandps          XMM3,   XMM4,   XMM3                ; XMM3 -> abs(norm(angle) - pi/2)
    mov             rax,    4560193532023345115
    vmovq           XMM4,   rax
    vpunpcklqdq     XMM4,   XMM4,   XMM4                ; XMM4 -> pi / 4
    vcmpps          XMM4,   XMM3,   XMM4, 01H           ; XMM4 -> sin_rectf (preserved)
    mov             rax,    4618102649103240164
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                ; XMM3 -> pi*3/4
    vcmpps          XMM5,   XMM1,   XMM3, 0EH           ; XMM5 -> cos_otherside (preserved)
    mov             rax,    4596222329050697691
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                ; XMM3 -> pi / 2
    vandps          XMM3,   XMM3,   XMM4                ; XMM3 -> sin_rectf ? pi / 2 : 0
    vsubps          XMM1,   XMM1,   XMM3
    mov             rax,    4632251126078050267
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                ; XMM3 -> pi
    vandps          XMM3,   XMM3,   XMM5
    vsubps          XMM1,   XMM1,   XMM3
    mov             rax,    8000000080000000H
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3                ; XMM3 -> 0x80000000
    vandps          XMM3,   XMM3,   XMM5
    vxorps          XMM1,   XMM1,   XMM3
    ; Pre-process of angle is finished
    mov             rax,    4575657222473777152
    vmovq           XMM0,   rax
    vpunpcklqdq     XMM0,   XMM0,   XMM0                ; XMM0 -> 1.f
    vmulps          XMM3,   XMM1,   XMM1                ; XMM3 -> x_sq (preserved)
    ; XMM0, XMM1, XMM2, XMM3, XMM4, and XMM5 are preserved
    mov             rax,    4539628425446424576
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6                        ; XMM6 -> 0.5f
    mov             rax,    4479580431832492715
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7                ; XMM7 -> 0.1666666667f
    vblendvps       XMM6,   XMM6,   XMM7, XMM4          ; XMM6 -> fact
    vmulps          XMM6,   XMM3,   XMM6                ; XMM6 -> x_term (preserved)
    vsubps          XMM0,   XMM0,   XMM6                ; XMM0 -> res (preserved)
    mov             rax,    4443551634805140139
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7                        ; XMM7 -> 0.0833333333f
    mov             rax,    4417130516412419277
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8                        ; XMM8 -> 0.05f
    vblendvps       XMM7,   XMM7,   XMM8, XMM4 ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7 ; Update x_term
    vaddps          XMM0,   XMM0,   XMM6 ; Update res
    mov             rax,    4397915159143155849
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.0333333333f
    mov             rax,    4378356668346600497
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8 ; XMM8 -> 0.0238095238f
    vblendvps XMM7, XMM7,   XMM8,   XMM4 ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7 ; Update x_term
    vsubps          XMM0,   XMM0,   XMM6 ; Update res
    mov             rax,    4364631413154269477
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.0178571429f
    mov             rax,    4351478041447468601
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8 ; XMM8 -> 0.0138888889f
    vblendvps XMM7, XMM7,   XMM8,   XMM4 ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7 ; Update x_term
    vaddps          XMM0,   XMM0,   XMM6 ; Update res
    mov             rax,    4338667803267959649
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.0111111111f
    mov             rax,    4329351263286522377
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8 ; XMM8 -> 0.0090909091f
    vblendvps       XMM7,   XMM7,   XMM8,   XMM4 ; XMM7 -> fact
    vmulps          XMM7,   XMM7,   XMM3
    vmulps          XMM6,   XMM6,   XMM7 ; Update x_term
    vsubps          XMM0,   XMM0,   XMM6 ; Update res
    ; XMM3, XMM6, and XMM7 are free
    mov             rax,    8000000080000000H
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3 ; XMM3 -> 0x80000000
    vxorps          XMM1,   XMM1,   XMM3 ; -norm(angle)
    mov             rax,    4575657222473777152
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6 ; XMM6 -> 1.f
    vblendvps XMM6, XMM6,   XMM1,   XMM4
    vmulps          XMM0,   XMM0,   XMM6 ; modified res by multiplying x when in sine case
    mov             rax,    100000001H
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1                ; XMM1 = <int>[1 1 1 1]
    cvtps2dq        XMM2,   XMM2                        ; XMM2 -> int(floor(period))
    pand            XMM2,   XMM1 ; XMM2 = sign of full_period_num
    pslld           XMM2,   1FH ; XMM2 = mask of sign inversion
    mov             rax,    8000000080000000H
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6 ; XMM6 -> 0x80000000
    vpand           XMM6,   XMM6,   XMM5
    vpxor           XMM2,   XMM2,   XMM6
    vpxor           XMM0,   XMM0,   XMM2 ; XMM0 -> masked inversed






    movups          XMM6, DWORD PTR [rsp - 16]
    movups          XMM7, DWORD PTR [rsp - 32]
    movups          XMM8, DWORD PTR [rsp - 48]
    pop rbx
    ret


_avx_cos_fp32x4@@16 ENDP





_avx_sin_fp32x4@@16 PROC
    mov             rax,    4596222329050697691 ; pi / 2
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1 ; XMM1 -> pi / 2
    vsubps          XMM0,   XMM1,   XMM0 ; XMM0 = pi/2 - angle




    call _avx_cos_fp32x4@@16
    ret

_avx_sin_fp32x4@@16 ENDP










_avx_cos_fp64x2@@16 PROC
    push rbx
    movups   DWORD PTR [rsp - 16],     XMM6
    movups   DWORD PTR [rsp - 32],     XMM7
    movups   DWORD PTR [rsp - 48],     XMM8
    mov             rax,    7fffffffffffffffH
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1 ; XMM1 -> inverted mask of a integer
    pand            XMM0,   XMM1 ; x = abs(x) (XMM0 -> abs(x))
    mov             rax,    4599405781057128579
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1
    vmulpd          XMM2,   XMM0,   XMM1 ; XMM2 -> x / pi
    vroundpd        XMM2,   XMM2,   01H ; XMM2 -> floor(x/pi) (preserved)
    mov             rax,    13837628693406821656
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1 ; XMM1 -> -pi
    vfmadd132pd     XMM1,   XMM0,   XMM2 ; XMM1 -> x - period * pi (normalized angle) (preserved)
    ; Angles normalized
    mov             rax,    4609753056924675352
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3 ; XMM3 -> pi / 2
    mov             rax,    7fffffffffffffffH
    vmovq           XMM4,   rax
    vpunpcklqdq     XMM4,   XMM4,   XMM4 ; XMM4 -> inverted mask of a integer
    vsubpd          XMM3,   XMM1,   XMM3 ; XMM3 -> norm(angle) - pi/2
    vandpd          XMM3,   XMM4,   XMM3 ; XMM3 -> abs(norm(angle) - pi/2)
    mov             rax,    4605249457297304856
    vmovq           XMM4,   rax
    vpunpcklqdq     XMM4,   XMM4,   XMM4 ; XMM4 -> pi / 4
    vcmppd          XMM4,   XMM3,   XMM4, 11H ; XMM4 -> sin_rectf (preserved)
    mov             rax,    4612488097114038738
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3 ; XMM3 -> pi*3/4
    vcmppd          XMM5,   XMM1,   XMM3, 1EH ; XMM5 -> cos_otherside (preserved)
    mov             rax,    4609753056924675352
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3 ; XMM3 -> pi / 2
    vandpd          XMM3,   XMM3,   XMM4 ; XMM3 -> sin_rectf ? pi / 2 : 0
    vsubpd          XMM1,   XMM1,   XMM3
    mov             rax,    4614256656552045848
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3 ; XMM3 -> pi
    vandpd          XMM3,   XMM3,   XMM5
    vsubpd          XMM1,   XMM1,   XMM3
    mov             rax,    8000000000000000H
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3 ; XMM3 -> 0x80000000
    vandpd          XMM3,   XMM3,   XMM5
    vxorpd          XMM1,   XMM1,   XMM3
    ; Pre-process of angle is finished
    mov             rax,    4607182418800017408
    vmovq           XMM0,   rax
    vpunpcklqdq     XMM0,   XMM0,   XMM0 ; XMM0 -> 1.f
    vmulpd          XMM3,   XMM1,   XMM1 ; XMM3 -> x_sq (preserved)
    ; XMM0, XMM1, XMM2, XMM3, XMM4, and XMM5 are preserved
    mov             rax,    4602678819172646912
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6 ; XMM6 -> 0.5f
    mov             rax,    4595172819793696087
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.1666666667f
    vblendvps XMM6, XMM6,   XMM7,   XMM4 ; XMM6 -> fact
    vmulpd          XMM6,   XMM3,   XMM6 ; XMM6 -> x_term (preserved)
    vsubpd          XMM0,   XMM0,   XMM6 ; XMM0 -> res (preserved)
    mov             rax,    4590669220166325587
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.0833333333f
    mov             rax,    4587366580439587226
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8 ; XMM8 -> 0.05f
    vblendvpd       XMM7,   XMM7,   XMM8,   XMM4 ; XMM7 -> fact
    vmulpd          XMM7,   XMM7,   XMM3
    vmulpd          XMM6,   XMM6,   XMM7 ; Update x_term
    vaddpd          XMM0,   XMM0,   XMM6 ; Update res
    mov             rax,    4584964660638322956
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.0333333333f
    mov             rax,    4582519849412036118
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8 ; XMM8 -> 0.0238095238f
    vblendvpd XMM7, XMM7,   XMM8,   XMM4 ; XMM7 -> fact
    vmulpd          XMM7,   XMM7,   XMM3
    vmulpd          XMM6,   XMM6,   XMM7 ; Update x_term
    vsubpd          XMM0,   XMM0,   XMM6 ; Update res
    mov             rax,    4580804192411133075
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.0178571429f
    mov             rax,    4579160021118600995
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8 ; XMM8 -> 0.0138888889f
    vblendvpd       XMM7,   XMM7,   XMM8,   XMM4 ; XMM7 -> fact
    vmulpd          XMM7,   XMM7,   XMM3
    vmulpd          XMM6,   XMM6,   XMM7 ; Update x_term
    vaddpd          XMM0,   XMM0,   XMM6 ; Update res
    mov             rax,    4577558741251091472
    vmovq           XMM7,   rax
    vpunpcklqdq     XMM7,   XMM7,   XMM7 ; XMM7 -> 0.0111111111f
    mov             rax,    4576394174074720926
    vmovq           XMM8,   rax
    vpunpcklqdq     XMM8,   XMM8,   XMM8 ; XMM8 -> 0.0090909091f
    vblendvpd       XMM7,   XMM7,   XMM8,   XMM4 ; XMM7 -> fact
    vmulpd          XMM7,   XMM7,   XMM3
    vmulpd          XMM6,   XMM6,   XMM7 ; Update x_term
    vsubpd          XMM0,   XMM0,   XMM6 ; Update res
    ; XMM3, XMM6, and XMM7 are free
    mov             rax,    8000000000000000H
    vmovq           XMM3,   rax
    vpunpcklqdq     XMM3,   XMM3,   XMM3 ; XMM3 -> 0x80000000
    vxorpd          XMM1,   XMM1,   XMM3 ; -norm(angle)
    mov             rax,    4607182418800017408
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6 ; XMM6 -> 1.f
    vblendvpd       XMM6,   XMM6,   XMM1, XMM4
    vmulpd          XMM0,   XMM0,   XMM6 ; modified res by multiplying x when in sine case
    mov             rax,    01H
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1 ; XMM1 = <int>[1 1 1 1]
    cvtpd2dq        XMM2,   XMM2 ; XMM2 -> int(floor(period))
    pshufd          XMM2,   XMM2,   216
    pand            XMM2,   XMM1 ; XMM2 = sign of full_period_num
    psllq           XMM2,   63 ; XMM2 = mask of sign inversion
    mov             rax,    8000000000000000H
    vmovq           XMM6,   rax
    vpunpcklqdq     XMM6,   XMM6,   XMM6 ; XMM6 -> 0x80000000
    vpand           XMM6,   XMM6,   XMM5
    vpxor           XMM2,   XMM2,   XMM6
    vpxor           XMM0,   XMM0,   XMM2 ; XMM0 -> masked inversed






    movups    XMM6, DWORD PTR [rsp - 16]
    movups    XMM7, DWORD PTR [rsp - 32]
    movups    XMM8, DWORD PTR [rsp - 48]
    pop             rbx
    ret

_avx_cos_fp64x2@@16 ENDP





_avx_sin_fp64x2@@16 PROC
    mov             rax,    4609753056924675352 ; pi / 2
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1 ; XMM1 -> pi / 2
    vsubpd          XMM0,   XMM1,   XMM0 ; XMM0 = pi/2 - angle




    call            _avx_cos_fp64x2@@16
    ret

_avx_sin_fp64x2@@16 ENDP









PUBLIC _avx_cos_fp32x4@@16
PUBLIC _avx_sin_fp32x4@@16
PUBLIC _avx_cos_fp64x2@@16
PUBLIC _avx_sin_fp64x2@@16
PUBLIC _avx_cos_fp32x8@@32

END
