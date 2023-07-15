/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDAEVENT_PACKAGE_CUH_
#define _CUDAEVENT_PACKAGE_CUH_


#include "../basic.h"
#include "cudaStream_package.h"


namespace decx
{
    class cuda_event;
}


class _DECX_API_ decx::cuda_event
{
private:
    cudaEvent_t _E;

public:
    int _event_flag;
    bool _is_occupied;

    cuda_event(const int flag);


    void detach();


    void attach();

    /* Call cudaStreamSynchronize() and the parameter is this->_S */
    void synchronize();


    void event_record(decx::cuda_stream* attached_stream);


    /* Return a referance of cudaStream_t object */
    cudaEvent_t& get_raw_event_ref();

    /* Return a pointer of cudaStream_t object */
    cudaEvent_t* get_raw_event_ptr();


    void release();
};




#endif