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

#ifdef __NASM__
default rel
#endif

#ifdef __MASM__
.DATA
#endif
#ifdef __NASM__
SECTION .data
#endif
    Abs_int_sign_DW     dq      7fffffffffffffffH
    Mask_MSB_DW         dq      8000000000000000H
    ONE_uint64          dq      01H
    ONE_fp64            dq      4607182418800017408   ; 1.f
    Pi_FP64             dq      4614256656552045848   ; 3.1415926536f
    ONE_over_PI_FP64    dq      4599405781057128579   ; 0.3183098862f
    Minus_Pi_FP64       dq      13837628693406821656   ; -3.1415926536f
    Halv_Pi_FP64        dq      4609753056924675352   ; 1.5707963268f
    Quarter_pi_FP64     dq      4605249457297304856   ; 0.7853981634f
    Three_4_Pi_FP64     dq      4612488097114038738   ; 2.3561944902f

    COS_TAYLOR_FP64     dq      4602678819172646912   ; 0.5
                        dq      4590669220166325587   ; 0.0833333333
                        dq      4584964660638322956   ; 0.0333333333
                        dq      4580804192411133075   ; 0.0178571429
                        dq      4577558741251091472   ; 0.0111111111

    SIN_TAYLOR_FP64     dq      4595172819793696087   ; 0.1666666667
                        dq      4587366580439587226   ; 0.05
                        dq      4582519849412036118   ; 0.0238095238f
                        dq      4579160021118600995   ; 0.0138888889f
                        dq      4576394174074720926   ; 0.0090909091f

#ifdef __NASM__
SECTION .text
#endif
#ifdef __MASM__
.CODE
#endif


#ifdef __NASM__
_avx_cos_fp64x2:
    push rbx
    movups          [rsp - 16],     XMM6
    movups          [rsp - 32],     XMM7
    movups          [rsp - 48],     XMM8
#endif
#ifdef __MASM__
_avx_cos_fp64x2@@16 PROC
    push rbx
    movups   DWORD PTR [rsp - 16],     XMM6
    movups   DWORD PTR [rsp - 32],     XMM7
    movups   DWORD PTR [rsp - 48],     XMM8
#endif
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
#ifdef __NASM__
    movups          XMM6,   [rsp - 16]
    movups          XMM7,   [rsp - 32]
    movups          XMM8,   [rsp - 48]
#endif
#ifdef __MASM__
    movups    XMM6, DWORD PTR [rsp - 16]
    movups    XMM7, DWORD PTR [rsp - 32]
    movups    XMM8, DWORD PTR [rsp - 48]
#endif
    pop             rbx
    ret
#ifdef __MASM__
_avx_cos_fp64x2@@16 ENDP
#endif


#ifdef __NASM__
_avx_cos_fp64x4:
    vmovupd         [rsp - 32],     YMM6
    vmovupd         [rsp - 64],     YMM7
    vmovupd         [rsp - 96],     YMM8

    vbroadcastsd    YMM1, QWORD [Abs_int_sign_DW]       ; YMM1 -> inverted mask of a integer
    vandpd          YMM0, YMM0, YMM1                    ; x = abs(x) (YMM0 -> abs(x))
    vbroadcastsd    YMM1, QWORD [ONE_over_PI_FP64]
    vmulpd          YMM2, YMM0, YMM1                    ; YMM2 -> x / pi
    vroundpd        YMM2, YMM2, 01H                     ; YMM2 -> floor(x/pi) (preserved)
    vbroadcastsd    YMM1, QWORD [Minus_Pi_FP64]         ; YMM1 -> -pi
    vfmadd132pd     YMM1, YMM0, YMM2                    ; YMM1 -> x - period * pi (normalized angle) (preserved)
    ; Angles normalized
    vbroadcastsd    YMM3, QWORD [Halv_Pi_FP64]          ; YMM3 -> pi / 2
    vbroadcastsd    YMM4, QWORD [Abs_int_sign_DW]       ; YMM4 -> 7fffffffH
    vsubpd          YMM3, YMM1, YMM3                    ; YMM3 -> norm(angle) - pi/2
    vandpd          YMM3, YMM4, YMM3                    ; YMM3 -> abs(norm(angle) - pi/2)
    vbroadcastsd    YMM4, QWORD [Quarter_pi_FP64]       ; YMM4 -> pi / 4
    vcmppd          YMM4, YMM3, YMM4, 01H               ; YMM4 -> sin_rectf (preserved)
    vbroadcastsd    YMM3, QWORD [Three_4_Pi_FP64]       ; YMM3 -> pi*3/4
    vcmppd          YMM5, YMM1, YMM3, 0EH               ; YMM5 -> cos_otherside (preserved)
    vbroadcastsd    YMM3, QWORD [Halv_Pi_FP64]          ; YMM3 -> Pi_FP32 / 2
    vandpd          YMM3, YMM3, YMM4                    ; YMM3 -> sin_rectf ? pi / 2 : 0
    vsubpd          YMM1, YMM1, YMM3
    vbroadcastsd    YMM3, QWORD [Pi_FP64]               ; YMM3 -> pi
    vandpd          YMM3, YMM3, YMM5
    vsubpd          YMM1, YMM1, YMM3
    vbroadcastsd    YMM3, QWORD [Mask_MSB_DW]           ; YMM3 -> 0x80000000
    vandpd          YMM3, YMM3, YMM5
    vxorpd          YMM1, YMM1, YMM3
    ; Pre-process of angle is finished
    vbroadcastsd    YMM0, QWORD [ONE_fp64]              ; YMM0 -> 1.f
    vmulpd          YMM3, YMM1, YMM1                    ; YMM3 -> x_sq (preserved)
    ; YMM0, YMM1, YMM2, YMM3, YMM4, and YMM5 are preserved
    vbroadcastsd    YMM6, QWORD [COS_TAYLOR_FP64]       ; YMM6 -> 0.5f
    vbroadcastsd    YMM7, QWORD [SIN_TAYLOR_FP64]       ; YMM7 -> 0.1666666667f
    vblendvpd       YMM6, YMM6, YMM7, YMM4              ; YMM6 -> fact
    vmulpd          YMM6, YMM3, YMM6                    ; YMM6 -> x_term (preserved)
    vsubpd          YMM0, YMM0, YMM6                    ; YMM0 -> res (preserved)
    vbroadcastsd    YMM7, QWORD [COS_TAYLOR_FP64 + 8]   ; YMM7 -> 0.0833333333f
    vbroadcastsd    YMM8, QWORD [SIN_TAYLOR_FP64 + 8]   ; YMM8 -> 0.05f
    vblendvpd       YMM7, YMM7, YMM8, YMM4              ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vaddpd          YMM0, YMM0, YMM6                    ; Update res
    vbroadcastsd    YMM7, QWORD [COS_TAYLOR_FP64 + 16]  ; YMM7 -> 0.0333333333f
    vbroadcastsd    YMM8, QWORD [SIN_TAYLOR_FP64 + 16]  ; YMM8 -> 0.0238095238f
    vblendvpd       YMM7, YMM7, YMM8, YMM4              ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vsubpd          YMM0, YMM0, YMM6                    ; Update res
    vbroadcastsd    YMM7, QWORD [COS_TAYLOR_FP64 + 24]  ; YMM7 -> 0.0178571429f
    vbroadcastsd    YMM8, QWORD [SIN_TAYLOR_FP64 + 24]  ; YMM8 -> 0.0138888889f
    vblendvpd YMM7, YMM7, YMM8, YMM4                    ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vaddpd          YMM0, YMM0, YMM6                    ; Update res
    vbroadcastsd    YMM7, QWORD [COS_TAYLOR_FP64 + 32]  ; YMM7 -> 0.0111111111f
    vbroadcastsd    YMM8, QWORD [SIN_TAYLOR_FP64 + 32]  ; YMM8 -> 0.0090909091f
    vblendvpd       YMM7, YMM7, YMM8, YMM4              ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vsubpd          YMM0, YMM0, YMM6                    ; Update res
    ; YMM3, YMM6, and YMM7 are free
    vbroadcastsd    YMM3, QWORD [Mask_MSB_DW]           ; YMM3 -> 0x80000000
    vxorpd          YMM1, YMM1, YMM3                    ; -norm(angle)
    vbroadcastsd    YMM6, QWORD [ONE_fp64]              ; YMM6 -> 1.f
    vblendvpd       YMM6, YMM6, YMM1, YMM4
    vmulpd          YMM0, YMM0, YMM6                    ; modified res by multiplying x when in sine case
    vbroadcastsd    YMM1, QWORD [ONE_uint64]            ; YMM1 = <int>[1 1 1 1]
    vcvtpd2dq       XMM2, YMM2                          ; YMM2 -> int(floor(period))
    vpmovsxdq       YMM2, XMM2,
    vandpd          YMM2, YMM2, YMM1                    ; YMM2 = sign of full_period_num
    vpsllq          YMM2, YMM2, 63                      ; YMM2 = mask of sign inversion
    vbroadcastsd    YMM6, QWORD [Mask_MSB_DW]           ; YMM6 -> 0x80000000
    vandpd          YMM6, YMM6, YMM5
    vxorpd          YMM2, YMM2, YMM6
    vxorpd          YMM0, YMM0, YMM2                    ; YMM0 -> masked inversed
    
    vmovupd         YMM6, [rsp - 32]
    vmovupd         YMM7, [rsp - 64]
    vmovupd         YMM8, [rsp - 96]
    ret
#endif
#ifdef __MASM__
_avx_cos_fp64x4@@32 PROC
    vmovupd         QWORD PTR [rsp - 32],     YMM6
    vmovupd         QWORD PTR [rsp - 64],     YMM7
    vmovupd         QWORD PTR [rsp - 96],     YMM8

    vbroadcastsd    YMM1, QWORD PTR [Abs_int_sign_DW]       ; YMM1 -> inverted mask of a integer
    vandpd          YMM0, YMM0, YMM1                    ; x = abs(x) (YMM0 -> abs(x))
    vbroadcastsd    YMM1, QWORD PTR [ONE_over_PI_FP64]
    vmulpd          YMM2, YMM0, YMM1                    ; YMM2 -> x / pi
    vroundpd        YMM2, YMM2, 01H                     ; YMM2 -> floor(x/pi) (preserved)
    vbroadcastsd    YMM1, QWORD PTR [Minus_Pi_FP64]         ; YMM1 -> -pi
    vfmadd132pd     YMM1, YMM0, YMM2                    ; YMM1 -> x - period * pi (normalized angle) (preserved)
    ; Angles normalized
    vbroadcastsd    YMM3, QWORD PTR [Halv_Pi_FP64]          ; YMM3 -> pi / 2
    vbroadcastsd    YMM4, QWORD PTR [Abs_int_sign_DW]       ; YMM4 -> 7fffffffH
    vsubpd          YMM3, YMM1, YMM3                    ; YMM3 -> norm(angle) - pi/2
    vandpd          YMM3, YMM4, YMM3                    ; YMM3 -> abs(norm(angle) - pi/2)
    vbroadcastsd    YMM4, QWORD PTR [Quarter_pi_FP64]       ; YMM4 -> pi / 4
    vcmppd          YMM4, YMM3, YMM4, 01H               ; YMM4 -> sin_rectf (preserved)
    vbroadcastsd    YMM3, QWORD PTR [Three_4_Pi_FP64]       ; YMM3 -> pi*3/4
    vcmppd          YMM5, YMM1, YMM3, 0EH               ; YMM5 -> cos_otherside (preserved)
    vbroadcastsd    YMM3, QWORD PTR [Halv_Pi_FP64]          ; YMM3 -> Pi_FP32 / 2
    vandpd          YMM3, YMM3, YMM4                    ; YMM3 -> sin_rectf ? pi / 2 : 0
    vsubpd          YMM1, YMM1, YMM3
    vbroadcastsd    YMM3, QWORD PTR [Pi_FP64]               ; YMM3 -> pi
    vandpd          YMM3, YMM3, YMM5
    vsubpd          YMM1, YMM1, YMM3
    vbroadcastsd    YMM3, QWORD PTR [Mask_MSB_DW]           ; YMM3 -> 0x80000000
    vandpd          YMM3, YMM3, YMM5
    vxorpd          YMM1, YMM1, YMM3
    ; Pre-process of angle is finished
    vbroadcastsd    YMM0, QWORD PTR [ONE_fp64]              ; YMM0 -> 1.f
    vmulpd          YMM3, YMM1, YMM1                    ; YMM3 -> x_sq (preserved)
    ; YMM0, YMM1, YMM2, YMM3, YMM4, and YMM5 are preserved
    vbroadcastsd    YMM6, QWORD PTR [COS_TAYLOR_FP64]       ; YMM6 -> 0.5f
    vbroadcastsd    YMM7, QWORD PTR [SIN_TAYLOR_FP64]       ; YMM7 -> 0.1666666667f
    vblendvpd       YMM6, YMM6, YMM7, YMM4              ; YMM6 -> fact
    vmulpd          YMM6, YMM3, YMM6                    ; YMM6 -> x_term (preserved)
    vsubpd          YMM0, YMM0, YMM6                    ; YMM0 -> res (preserved)
    vbroadcastsd    YMM7, QWORD PTR [COS_TAYLOR_FP64 + 8]   ; YMM7 -> 0.0833333333f
    vbroadcastsd    YMM8, QWORD PTR [SIN_TAYLOR_FP64 + 8]   ; YMM8 -> 0.05f
    vblendvpd       YMM7, YMM7, YMM8, YMM4              ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vaddpd          YMM0, YMM0, YMM6                    ; Update res
    vbroadcastsd    YMM7, QWORD PTR [COS_TAYLOR_FP64 + 16]  ; YMM7 -> 0.0333333333f
    vbroadcastsd    YMM8, QWORD PTR [SIN_TAYLOR_FP64 + 16]  ; YMM8 -> 0.0238095238f
    vblendvpd       YMM7, YMM7, YMM8, YMM4              ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vsubpd          YMM0, YMM0, YMM6                    ; Update res
    vbroadcastsd    YMM7, QWORD PTR [COS_TAYLOR_FP64 + 24]  ; YMM7 -> 0.0178571429f
    vbroadcastsd    YMM8, QWORD PTR [SIN_TAYLOR_FP64 + 24]  ; YMM8 -> 0.0138888889f
    vblendvpd YMM7, YMM7, YMM8, YMM4                    ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vaddpd          YMM0, YMM0, YMM6                    ; Update res
    vbroadcastsd    YMM7, QWORD PTR [COS_TAYLOR_FP64 + 32]  ; YMM7 -> 0.0111111111f
    vbroadcastsd    YMM8, QWORD PTR [SIN_TAYLOR_FP64 + 32]  ; YMM8 -> 0.0090909091f
    vblendvpd       YMM7, YMM7, YMM8, YMM4              ; YMM7 -> fact
    vmulpd          YMM7, YMM7, YMM3
    vmulpd          YMM6, YMM6, YMM7                    ; Update x_term
    vsubpd          YMM0, YMM0, YMM6                    ; Update res
    ; YMM3, YMM6, and YMM7 are free
    vbroadcastsd    YMM3, QWORD PTR [Mask_MSB_DW]           ; YMM3 -> 0x80000000
    vxorpd          YMM1, YMM1, YMM3                    ; -norm(angle)
    vbroadcastsd    YMM6, QWORD PTR [ONE_fp64]              ; YMM6 -> 1.f
    vblendvpd       YMM6, YMM6, YMM1, YMM4
    vmulpd          YMM0, YMM0, YMM6                    ; modified res by multiplying x when in sine case
    vbroadcastsd    YMM1, QWORD PTR [ONE_uint64]            ; YMM1 = <int>[1 1 1 1]
    vcvtpd2dq       XMM2, YMM2                          ; YMM2 -> int(floor(period))
    vpmovsxdq       YMM2, XMM2,
    vandpd          YMM2, YMM2, YMM1                    ; YMM2 = sign of full_period_num
    vpsllq          YMM2, YMM2, 63                      ; YMM2 = mask of sign inversion
    vbroadcastsd    YMM6, QWORD PTR [Mask_MSB_DW]           ; YMM6 -> 0x80000000
    vandpd          YMM6, YMM6, YMM5
    vxorpd          YMM2, YMM2, YMM6
    vxorpd          YMM0, YMM0, YMM2                    ; YMM0 -> masked inversed

    vmovupd         YMM6, QWORD PTR [rsp - 32]
    vmovupd         YMM7, QWORD PTR [rsp - 64]
    vmovupd         YMM8, QWORD PTR [rsp - 96]
    ret
_avx_cos_fp64x4@@32 ENDP
#endif

#ifdef __NASM__
_avx_sin_fp64x2:
#endif
#ifdef __MASM__
_avx_sin_fp64x2@@16 PROC
#endif
    mov             rax,    4609753056924675352 ; pi / 2
    vmovq           XMM1,   rax
    vpunpcklqdq     XMM1,   XMM1,   XMM1 ; XMM1 -> pi / 2
    vsubpd          XMM0,   XMM1,   XMM0 ; XMM0 = pi/2 - angle
#ifdef __NASM__
    call            _avx_cos_fp64x2
#endif
#ifdef __MASM__
    call            _avx_cos_fp64x2@@16
#endif
    ret
#ifdef __MASM__
_avx_sin_fp64x2@@16 ENDP
#endif


#ifdef __NASM__
_avx_sin_fp64x4:
#endif
#ifdef __MASM__
_avx_sin_fp64x4@@32 PROC
#endif
#ifdef __MASM__
    vbroadcastsd    YMM1, QWORD PTR [Halv_Pi_FP64]
#endif
#ifdef __NASM__
    vbroadcastsd    YMM1, QWORD [Halv_Pi_FP64]
#endif
    vsubpd          YMM0,   YMM1,   YMM0 ; XMM0 = pi/2 - angle
#ifdef __NASM__
    call            _avx_cos_fp64x4
#endif
#ifdef __MASM__
    call            _avx_cos_fp64x4@@32
#endif
    ret
#ifdef __MASM__
_avx_sin_fp64x4@@32 ENDP
#endif


#ifdef __NASM__
GLOBAL _avx_cos_fp64x2
GLOBAL _avx_sin_fp64x2
GLOBAL _avx_cos_fp64x4
GLOBAL _avx_sin_fp64x4
#endif
#ifdef __MASM__
PUBLIC _avx_cos_fp64x2@@16
PUBLIC _avx_sin_fp64x2@@16
PUBLIC _avx_cos_fp64x4@@32
PUBLIC _avx_sin_fp64x4@@32


END
#endif
