/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "DecxStream.h"


decx::_decx_stream::_decx_stream()
{
    this->_stream_id = decx::async::async_engine.add_stream();
    this->_async_caller = decx::async::async_engine[_stream_id];

    this->_idle = false;
}



uint decx::_decx_stream::Get_ID()
{
    return this->_stream_id;
}



void decx::_decx_stream::execute()
{
    decx::async::_ATE* _current_ATE_ptr = this->_async_caller;
    _current_ATE_ptr->_all_done = false;

    _current_ATE_ptr->_cv.notify_one();
}



void decx::_decx_stream::synchronize()
{
    decx::async::_ATE* _current_ATE_ptr = decx::async::async_engine[this->_stream_id];
    _current_ATE_ptr->_all_done = false;

    _current_ATE_ptr->_cv.notify_one();
    std::unique_lock<std::mutex> lock(decx::async::async_engine._mtx_main_thread);
    if (!_current_ATE_ptr->_all_done) {
        decx::async::async_engine._cv_main_thread.wait(lock);
    }
}


de::DH* decx::_decx_stream::Get_last_handle()
{
    return &this->_last_handle;
}



_DECX_API_ de::DecxStream& de::CreateAsyncStreamRef()
{
    return *(new decx::_decx_stream);
}


_DECX_API_ de::DecxStream* de::CreateAsyncStreamPtr()
{
    return (new decx::_decx_stream);
}


_DECX_API_ void de::DestroyDecxStream(de::DecxStream& stream)
{
    decx::_decx_stream* _stream = reinterpret_cast<decx::_decx_stream*>(&stream);
    _stream->_async_caller->_idle = true;
}



_DECX_API_ void de::Execute_All_Streams()
{
    for (int i = 0; i < decx::async::async_engine.get_current_stream_num(); ++i) {
        decx::async::_ATE* _current_ATE_ptr = decx::async::async_engine[i];
        _current_ATE_ptr->_all_done = false;

        _current_ATE_ptr->_cv.notify_one();
    }
}



_DECX_API_ void de::Global_Synchronize()
{
    for (int i = 0; i < decx::async::async_engine.get_current_stream_num(); ++i) {
        decx::async::_ATE* _current_ATE_ptr = decx::async::async_engine[i];
        if (!_current_ATE_ptr->_all_done) {
            std::unique_lock<std::mutex> lock(decx::async::async_engine._mtx_main_thread);
            decx::async::async_engine._cv_main_thread.wait(lock);
        }
    }
}