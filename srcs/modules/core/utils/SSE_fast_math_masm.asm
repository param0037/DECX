; ---------------------------------------------------------------------
; Author : Wayne Anderson
; Date   : 2021.04.16
; ---------------------------------------------------------------------
; This is a part of the open source program named "DECX", copyright c Wayne,
; 2021.04.16, all right reserved.
; More information please visit https://github.com/param0037/DECX


.DATA
ALIGN 16
    sign_i32_masks  dd      7fffffffH, 7fffffffH, 7fffffffH, 7fffffffH
    ONE_over_PI     dd      0.3183098862f,  0.3183098862f,  0.3183098862f,  0.3183098862f
    Minus_Pis       dd      -3.1415926536f, -3.1415926536f, -3.1415926536f, -3.1415926536f
    ONES_fp32       dd      1.f,            1.f,            1.f,            1.f
    ONE_over_2      dd      0.5f,           0.5f,           0.5f,           0.5f
    ONE_over_6      dd      0.1666666667f,  0.1666666667f,  0.1666666667f,  0.1666666667f
    ONE_over_12     dd      0.0833333333f,  0.0833333333f,  0.0833333333f,  0.0833333333f
    ONE_over_20     dd      0.05f,          0.05f,          0.05f,          0.05f
    ONE_over_30     dd      0.0333333333f,  0.0333333333f,  0.0333333333f,  0.0333333333f
    ONE_over_42     dd      0.0238095238f,  0.0238095238f,  0.0238095238f,  0.0238095238f
    ONE_over_56     dd      0.0178571429f,  0.0178571429f,  0.0178571429f,  0.0178571429f
    ONE_over_72     dd      0.0138888889f,  0.0138888889f,  0.0138888889f,  0.0138888889f
    ONE_over_90     dd      0.0111111111f,  0.0111111111f,  0.0111111111f,  0.0111111111f
    ONE_over_110    dd      0.0090909091f,  0.0090909091f,  0.0090909091f,  0.0090909091f
    ONE_over_156    dd      0.0064102564f,  0.0064102564f,  0.0064102564f,  0.0064102564f
    ONES_int32      dd      01H,            01H,            01H,            01H


; __vectorcall convention
.CODE
fast_mm_cos_ps@@16 PROC

    movaps          XMM1,   [sign_i32_masks]    ; XMM1 = [mask]
    pand            XMM0,   XMM1                ; x = abs(x) (XMM0 -> abs(x))

    movaps          XMM1,   [ONE_over_PI]       ; XMM1 = [1/Pi 1/Pi 1/Pi 1/Pi]
    mulps           XMM1,   XMM0                ; XMM1 = |x| / Pi

    roundps         XMM1,   XMM1,   08H         ; XMM1 -> round off full_period_num

    cvtps2dq        XMM1,   XMM1                ; XMM1 = (int)XMM1 = full_period_num    (preserved)
    cvtdq2ps        XMM2,   XMM1                ; XMM2 = floor(|x| / Pi)
    movaps          XMM3,   [Minus_Pis]         ; XMM3 = [-Pi -Pi -Pi -Pi]
    vfmadd132ps     XMM2,   XMM0,   XMM3        ; XMM2 = Normalized input [x]

    mulps           XMM2,   XMM2                ; XMM2 = Norm([x])^2 = x_sq             (preserved)

    movaps          XMM3,   [ONE_over_2]        ; XMM3 = [.5 .5 .5 .5]
    mulps           XMM3,   XMM2                ; XMM3 = x_term = x_sq / 2              (preserved)

    movaps          XMM0,   [ONES_fp32]         ; XMM0 = <float>[1 1 1 1]
    subps           XMM0,   XMM3                ; XMM0 = res = 1 - x_sq / 2

    movaps          XMM4,   [ONE_over_12]       ; XMM4 = [1/12 1/12 1/12 1/12]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 12
    mulps           XMM3,   XMM4                ; x_term updated
    addps           XMM0,   XMM3                ; res updated

    movaps          XMM4,   [ONE_over_30]       ; XMM4 = [1/30 1/30 1/30 1/30]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 30
    mulps           XMM3,   XMM4                ; x_term updated
    subps           XMM0,   XMM3                ; res updated

    movaps          XMM4,   [ONE_over_56]       ; XMM4 = [1/56 1/56 1/56 1/56]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 56
    mulps           XMM3,   XMM4                ; x_term updated
    addps           XMM0,   XMM3                ; res updated

    movaps          XMM4,   [ONE_over_90]       ; XMM4 = [1/90 1/90 1/90 1/90]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 90
    mulps           XMM3,   XMM4                ; x_term updated
    addps           XMM0,   XMM3                ; res updated
    ; XMM2~4 are freed
    
    movaps          XMM2,   [ONES_int32]        ; XMM2 = <int>[1 1 1 1]
    pand            XMM1,   XMM2                ; XMM1 = sign of full_period_num
    pslld           XMM1,   1FH                 ; XMM1 = mask of sign inversion
    pxor            XMM0,   XMM1                ; XMM3 -> masked inversed

    ret
    
fast_mm_cos_ps@@16 ENDP

PUBLIC fast_mm_cos_ps@@16

fast_mm_sin_ps@@16 PROC

    movaps          XMM1,   [ONE_over_PI]       ; XMM2 = [1/Pi 1/Pi 1/Pi 1/Pi]
    movaps          XMM2,   [ONE_over_2]        ; XMM3 = [.5 .5 .5 .5]
    vfmadd132ps     XMM1,   XMM2,   XMM0        ; XMM1 = __x/Pi + 1/2 = (__x + Pi/2)/Pi
    
    roundps         XMM1,   XMM1,   08H         ; XMM1 -> round off full_period_num

    cvtps2dq        XMM1,   XMM1                ; XMM1 = (int)XMM1 = full_period_num    (preserved)
    cvtdq2ps        XMM2,   XMM1                ; XMM2 = floor((__x + Pi/2)/Pi)
    movaps          XMM3,   [Minus_Pis]         ; XMM3 = [-Pi -Pi -Pi -Pi]
    vfmadd132ps     XMM2,   XMM0,   XMM3        ; XMM2 = Normalized input [x]
    movaps          XMM0,   XMM2                ; Copy normed from XMM2 to XMM0

    mulps           XMM2,   XMM2                ; XMM2 = Norm([x])^2 = x_sq             (preserved)

    movaps          XMM3,   [ONE_over_6]        ; XMM3 = [1/6 1/6 1/6 1/6]
    mulps           XMM3,   XMM2                ; XMM3 = x_sq / 6
    mulps           XMM3,   XMM0                ; XMM3 = x_term = x_sq * normed / 6     (preserved)

    subps           XMM0,   XMM3                ; XMM0 = res = normed - x^3 / 6

    movaps          XMM4,   [ONE_over_20]       ; XMM4 = [1/20 1/20 1/20 1/20]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 20
    mulps           XMM3,   XMM4                ; x_term updated
    addps           XMM0,   XMM3                ; res updated

    movaps          XMM4,   [ONE_over_42]       ; XMM4 = [1/42 1/42 1/42 1/42]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 42
    mulps           XMM3,   XMM4                ; x_term updated
    subps           XMM0,   XMM3                ; res updated

    movaps          XMM4,   [ONE_over_72]       ; XMM4 = [1/72 1/72 1/72 1/72]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 72
    mulps           XMM3,   XMM4                ; x_term updated
    addps           XMM0,   XMM3                ; res updated

    movaps          XMM4,   [ONE_over_110]      ; XMM4 = [1/110 1/110 1/110 1/110]
    mulps           XMM4,   XMM2                ; XMM4 = x_sq / 110
    mulps           XMM3,   XMM4                ; x_term updated
    subps           XMM0,   XMM3                ; res updated

    ; movaps          XMM4,   [ONE_over_156]      ; XMM4 = [1/156 1/156 1/156 1/156]
    ; mulps           XMM4,   XMM2                ; XMM4 = x_sq / 156
    ; mulps           XMM3,   XMM4                ; x_term updated
    ; addps           XMM0,   XMM3                ; res updated
    ; XMM2~4 are freed

    movaps          XMM2,   [ONES_int32]        ; XMM2 = <int>[1 1 1 1]
    pabsd           XMM1,   XMM1                ; XMM1 = |full_period_num|
    pand            XMM1,   XMM2                ; XMM1 = sign of full_period_num
    pslld           XMM1,   1FH                 ; XMM1 = mask of sign inversion
    pxor            XMM0,   XMM1                ; XMM3 -> masked inversed

    ret
    
fast_mm_sin_ps@@16 ENDP

PUBLIC fast_mm_sin_ps@@16

END
