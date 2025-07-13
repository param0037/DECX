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

/*
* Header for resource manager.
* 
* In DECX, many algorithms need to be planned ahead, i.e. FFT, Im2col, filter, etc.
* To prevent repeated planning whenever the alogrithm being executed, the planned 
* results will be stored as global variables. However, the problem is, once the algorithm
* being called once, the planned results will never be released, until the end of the 
* program. To solve this, resource manager is used. Each planned result mentioned
* above is regarded as a resource and managed by the resource manager thread running
* by library (lib)DECX_core_CPU.
* 
* For each resource, the attribute _lifespan controls its exsisting time after the last
* used. If the resource is not being used and it exceeds its lifespan, the manager will
* call the deconstructor callback of the resource and then delete it.
*/

#ifndef _DECX_RESOURCE_H_
#define _DECX_RESOURCE_H_

#include <basic.h>
#include <Array/Dynamic_Array.h>
#include <Concurrent/task_handle.h>

namespace decx
{
    typedef void (*res_release_fn)(void*);

    class Resource;

    class ResMgr;

    extern decx::ResMgr* _res_mgr;
    

    _DECX_API_ uint64_t ResourceCheckIn(void** exposed_ptr, const time_t lifespan_sec, res_release_fn _decon);


    _DECX_API_ void ResourceLock(const uint64_t res_id);


    _DECX_API_ void ResourceUnlock(const uint64_t res_id);


    _DECX_API_ void ResourceCheckOut(const uint64_t res_id);
}


class decx::ResMgr
{
private:
    std::condition_variable                     _cv;
    std::mutex                                  _mtx;
    uint64_t                                    _last_res_num;
    bool                                        _run;
    decx::utils::Dynamic_Array<decx::Resource>  _res_arr;
    decx::core::TaskHandle_t                    _task;
    time_t                                      _shortest_wait_period;

private:
    _THREAD_FUNCTION_ static void __ResMgrTask(decx::ResMgr*);

    struct _wait_pred
    {
        decx::ResMgr* _outer_info;

        bool operator() () {
            return (_outer_info->_last_res_num != _outer_info->_res_arr.size()) && 
                   (this->_outer_info->_res_arr.size() != 0);
        }
    }_wp;

public:
    ResMgr();


    uint64_t checkin(void** exposed_ptr, const time_t lifespan, res_release_fn _decon);


    void checkout(const uint64_t res_id);


    void lock_resource(const uint64_t res_id);


    void unlock_resource(const uint64_t res_id);


    ~ResMgr();
};


class decx::Resource
{
private:
    void** _exposed_ptr;

    time_t _last_used_instant;

    time_t _lifespan_sec;

    res_release_fn _deconstructor_callback;

    bool _occupied;

public:
    Resource();


    Resource(void** exposed_ptr, const time_t lifespan_sec, res_release_fn rel_fn);


    bool exceeded_lifespan(const time_t now) const;


    time_t get_last_used_instant() const;


    time_t get_lifespan() const;


    void lock();


    void unlock();


    bool Delete();
};


#endif
