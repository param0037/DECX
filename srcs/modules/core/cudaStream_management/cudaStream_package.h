/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CUDASTREAM_PACKAGE_CUH_
#define _CUDASTREAM_PACKAGE_CUH_


#include "../basic.h"

namespace decx
{
    class cuda_stream;
}


class decx::cuda_stream
{
private:
    cudaStream_t _S;

public:
    int _stream_flag;
    bool _is_occupied;

    cuda_stream(const int flag);


    void detach();


    void attach();

    /* Call cudaStreamSynchronize() and the parameter is this->_S */
    void this_stream_sync();

    /* Return a referance of cudaStream_t object */
    cudaStream_t& get_raw_stream_ref();

    /* Return a pointer of cudaStream_t object */
    cudaStream_t* get_raw_stream_ptr();


    void release();


    ~cuda_stream() {}
};




#endif