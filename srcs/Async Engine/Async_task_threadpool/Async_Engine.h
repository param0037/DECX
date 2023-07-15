/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _ASYNC_ENGINE_H_
#define _ASYNC_ENGINE_H_


#include "../../modules/core/basic.h"
#include "../../modules/core/thread_management/task_queue.h"
#include "../DecxStream/DecxStream.h"
#include "../Async_task_threadpool/async_task_executor.h"

#ifdef _DECX_ASYNC_CORE_
namespace decx
{
    namespace async {
        class Async_Engine;
    }
}
#endif

namespace decx {
    namespace async {
        template <class FuncType, class ...Args>
        static void register_async_task(const size_t id, FuncType&& f, Args&& ...args);
    }
}



namespace decx
{
    namespace async 
    {
#ifdef _DECX_ASYNC_CORE_
        extern decx::async::Async_Engine async_engine;
#endif

        _DECX_API_ decx::async::_ATE* _get_ATE_by_id(const uint64_t id);


        _DECX_API_ uint64_t AsyncEngine_add_stream();
    }
}

#ifdef _DECX_ASYNC_CORE_


class decx::async::Async_Engine
{
    friend _DECX_API_ decx::async::_ATE* _get_ATE_by_id(const uint64_t id);
private:
    std::thread* _thread_list;

    decx::async::async_task_executor* _task_schd;

    uint current_stream_num;

    /**
    * @brief : The actual capacity (in physical memory) of the two lists above. When reach the maximum,
    * deallocate the current space and apply for a new and longer space.
    * NOT IN BYTES
    */
    uint actual_capacity;

    /**
    * @brief : Find available stream(not occupied ones) for user, and label the found
    * stream as occupied
    * @param : The pointer of the ID of the found stream
    * @return : True if found, False if not found
    */
    bool find_available_stream(int* _ID);


    void resize();

public:

    std::mutex _mtx_main_thread;
    std::condition_variable _cv_main_thread;

    Async_Engine();


    uint add_stream();


    void shutdown();


    decx::async::_ATE* operator[](const size_t dex);


    uint get_current_stream_num();


    ~Async_Engine();
};



#endif


template <class FuncType, class ...Args>
void decx::async::register_async_task(const uint64_t id, FuncType&& f, Args&& ...args)
{
    decx::async::async_task_executor* _tq = decx::async::_get_ATE_by_id(id);

    _tq->_task_queue.emplace_back(std::bind(std::forward<FuncType>(f), std::forward<Args>(args)...));

    ++_tq->_task_num;
}


#endif