# Element-wise Operators Conventions

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
