/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/

#include "../../../core/basic.h"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../classes/type_info.h"


namespace decx
{
    namespace scan
    {
        class cuda_scan1D_config;


        // 1D
        void cuda_scan1D_fp32_caller_Async(const decx::scan::cuda_scan1D_config* _config, decx::cuda_stream* S);


        void cuda_scan1D_u8_i32_caller_Async(const decx::scan::cuda_scan1D_config* _config, decx::cuda_stream* S);


        void cuda_scan1D_fp16_caller_Async(const decx::scan::cuda_scan1D_config* _config, decx::cuda_stream* S);
    }
}


class decx::scan::cuda_scan1D_config
{
    /*
    * The type of data in the input array, usually fp32, uint8 and fp16
    * Indicated by decx::_DATA_TYPE_FLAGES_
    */
    de::_DATA_TYPES_FLAGS_ _data_type_in;

    /*
    * The type of data in the output array, usually fp32 and int32
    * Indicated by decx::_DATA_TYPE_FLAGES_
    * 
    * specialised:
    *   fp32    -> fp32
    *   uint8   -> int32
    *   fp16    -> fp32
    */
    de::_DATA_TYPES_FLAGS_ _data_type_out;

    /*
    * The pointers of source array, destinated array and status array
    */
    decx::PtrInfo<void> _dev_src, _dev_status, _dev_dst, _dev_tmp;

    /*
    * The length of the processed area. Measured in scale of element
    */
    uint64_t _length;

    /*
    * Required number of the CUDA thread blocks
    */
    uint64_t _block_num;

    /*
    * scan mode
    * Indicated by decx::scan::_warp_scan_mode
    * 
    * specialised:
    *   1. exclusive_scan
    *   2. inclusive_scan
    */
    int _scan_mode;

public:
    cuda_scan1D_config() {}

    /*
    * construct cuda_scan1D_config by just indicating the input & output types. Device memory is
    * allocated during the construction
    */
    template <uint32_t _align, typename _type_in, typename _type_out = _type_in>
    void generate_scan_config(const uint64_t _proc_length, decx::cuda_stream* S, const int scan_mode);


    template <uint32_t _align, typename _type_in, typename _type_out = _type_in>
    void generate_scan_config(decx::PtrInfo<void> dev_src, decx::PtrInfo<void> dev_dst, const uint64_t _proc_length, 
        decx::cuda_stream* S, const int scan_mode);


    uint64_t get_proc_length() const;
    uint64_t get_block_num() const;
    int get_scan_mode() const;


    void* get_raw_dev_ptr_src() const;

    void* get_raw_dev_ptr_dst() const;

    void* get_raw_dev_ptr_status() const;

    void* get_raw_dev_ptr_tmp() const;

    template <typename _src_type>
    void release_buffer(const bool _have_dev_classes);
};



namespace decx
{
    namespace scan
    {
        class cuda_scan2D_config;


        // 2D
        template <bool _only_scan_h>
        void cuda_scan2D_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);


        template <bool _only_scan_h>
        void cuda_scan2D_u8_i32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);

        template <bool _only_scan_h>
        void cuda_scan2D_fp16_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);

        //  vertically scan
        void cuda_scan2D_v_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);


        void cuda_scan2D_v_fp16_fp32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);


        void cuda_scan2D_v_u8_i32_caller_Async(const decx::scan::cuda_scan2D_config* _config, decx::cuda_stream* S);
    }
}


class decx::scan::cuda_scan2D_config
{
private:
    /*
    * The type of data in the input array, usually fp32, uint8 and fp16
    * Indicated by decx::_DATA_TYPE_FLAGES_
    */
    de::_DATA_TYPES_FLAGS_ _data_type_in;

    /*
    * The type of data in the output array, usually fp32 and int32
    * Indicated by decx::_DATA_TYPE_FLAGES_
    *
    * specialised:
    *   fp32    -> fp32
    *   uint8   -> int32
    *   fp16    -> fp32
    */
    de::_DATA_TYPES_FLAGS_ _data_type_out;

    int _scan_mode;

    /*
    * The pointers of source array, destinated array and status array
    * 
    * dev_tmp is used only when fp16 mid-results are involved in integrating uint8_t
    */
    decx::Ptr2D_Info<void> _dev_src, _dev_dst, _dev_tmp;
    decx::PtrInfo<void> _dev_status;

    dim3 _scan_h_grid, _scan_v_grid;

public:
    cuda_scan2D_config() {}

    /*
    * construct cuda_scan1D_config by just indicating the input & output types. Device memory is
    * allocated during the construction
    * 
    * @param proc_dims : ~.x -> width of the processing area, mesaured in scale of 1 element
    *                    ~.y -> height of the processing area, meausred in scale of 1 element
    * 
    */
    template <typename _type_in, typename _type_out = _type_in>
    void generate_scan_config(const uint2 _proc_dims, decx::cuda_stream* S, const int scan_mode, 
        const bool _is_full_scan);

    template <typename _type_in, typename _type_out = _type_in>
    void generate_scan_config(decx::Ptr2D_Info<void> dev_src, decx::Ptr2D_Info<void> dev_dst, const uint2 _proc_dims, 
        decx::cuda_stream* S, const int scan_mode, const bool _is_full_scan);


    int get_scan_mode() const;

    decx::Ptr2D_Info<void> get_raw_dev_ptr_src() const;

    decx::Ptr2D_Info<void> get_raw_dev_ptr_dst() const;

    decx::PtrInfo<void> get_raw_dev_ptr_status() const;


    decx::Ptr2D_Info<void> get_raw_dev_ptr_tmp() const;

    dim3 get_scan_h_grid() const;

    dim3 get_scan_v_grid() const;

    template <typename _Type_src>
    /*
    * @param : _refer_dev_classes : Do dev_src and dev_dst refer to exsiting classes (from users)
    */
    void release_buffer(const bool _refer_dev_classes);
};