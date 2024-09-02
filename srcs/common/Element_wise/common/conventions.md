# Element-wise Operators Conventions

## Naming rules
1. For operators or kernels holding 1 variable input, 1 constant input, and 1 output, named by suffix "<font color="yellow">_VCO</font>".
2. For operators or kernels holding 2 variable inputs, and 1 output, named by suffix "<font color="yellow">_VVO</font>".
3. For operators or kernels holding 1 variable input, and 1 output, named by suffix "<font color="yellow">_VO</font>".

## 1D operators
### a. Unary operators
1. CPU<br>
defined with type of 
    ```c
    typedef void(unary_1D_ops)(const _type_in* src, _type_out* dst, const uint64_t _proc_len_v, additional_args...);
    ```
_proc_len_v is the processing length, but in stride of SIMD vector length.

2. CUDA<br>

### b. Binary operators
1. CPU<br>
    defined with type of 
    ```c
    typedef void(binary_1D_ops)(const _type_in* src1, const _type_in* src2, _type_out* dst, const uint64_t _proc_len_v, additional_args...);
    ```

## 2D opeerators
### a. Unary operators
1. CPU<br>
    defined with type of
    ```c
    typedef void(unary_2D_ops)(const _type_in* src, _type_out* dst, const uint2 _proc_dims_v, const uint32_t Wsrc, const uint32_t Wdst);
    ```
### b. Binary operators
1. CPU<br>
    ```c
    typedef void(binary_2D_ops)(const _type_in* src1, const _type_in* src2, _type_out* dst, const uint2 _proc_dims_v, const uint32_t Wsrc, const uint32_t Wdst);
    ```
