/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DECX_STREAM_H_
#define _DECX_STREAM_H_


#include "../../modules/core/basic.h"
#include "../Async_task_threadpool/async_task_executor.h"
#include "../Async_task_threadpool/Async_Engine.h"


namespace de
{
    class _DECX_API_ DecxStream
    {
    public:
        virtual uint Get_ID() = 0;


        virtual void execute() = 0;


        virtual void synchronize() = 0;


        virtual de::DH* Get_last_handle() = 0;
    };


    _DECX_API_ de::DecxStream& CreateAsyncStreamRef();


    _DECX_API_ de::DecxStream* CreateAsyncStreamPtr();


    _DECX_API_ void DestroyDecxStream(de::DecxStream& stream);


    _DECX_API_ void Execute_All_Streams();


    _DECX_API_ void Global_Synchronize();
}


#ifdef _DECX_ASYNC_CORE_
namespace decx
{
    class _decx_stream : public de::DecxStream
    {
    public:
        decx::async::async_task_executor* _async_caller;

        de::DH _last_handle;

        uint64_t _stream_id;
        bool _idle;

        _decx_stream();


        virtual uint Get_ID();


        virtual void execute();


        virtual void synchronize();


        virtual de::DH* Get_last_handle();
    };
}
#endif

#endif