/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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

#include "../basic.h"
#include "../utils/Dynamic_Array.h"

namespace decx
{
    struct ResourceHandle;

    typedef void (*res_release_fn)(void*);

#ifdef _DECX_CORE_CPU_
    class Resource;

    class ResMgr;

    extern decx::ResMgr* _res_mgr;
#endif

    _DECX_API_ uint64_t ResourceCheckIn(void** exposed_ptr, const time_t lifespan_sec, res_release_fn _decon);


    _DECX_API_ void ResourceLock(const uint64_t res_id);


    _DECX_API_ void ResourceUnlock(const uint64_t res_id);


    _DECX_API_ void ResourceCheckOut(const uint64_t res_id);
}


#ifdef _DECX_CORE_CPU_


class decx::ResMgr
{
private:
    std::condition_variable _cv;

    std::mutex _mtx;

    uint64_t _last_res_num;

    bool _run;

    decx::utils::Dynamic_Array<decx::Resource> _res_arr;

    std::thread* _mgr_thread;

    time_t _shortest_wait_period;

    void _mgr_task();

    struct _wait_pred
    {
        decx::ResMgr* _outer_info;

        bool operator() () {
            return _outer_info->_last_res_num != _outer_info->_res_arr.size()
                && this->_outer_info->_res_arr.size() != 0;
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

#endif  // #ifdef _DECX_CORE_CPU_


struct decx::ResourceHandle
{
    void* _res_ptr;
    uint64_t _res_id;

    ResourceHandle()
    {
        this->_res_id = 0;
        this->_res_ptr = NULL;
    }

    /**
    * @brief :          Register (or checkin) a resource.
    * @param res_ptr :  The resource raw pointer.
    * @param lifespan : The lifespan of the resource, in second.
    * @param _decon :   The callback function when checkout the resource. 
    *                   Note : This callback function must be in type void func(type*).
    *                   If the function is a member function of a class, plase define it as static.
    */
    template <class _decon_type>
    void RegisterResource(void* res_ptr, const time_t lifespan, _decon_type* _decon = NULL)
    {
        this->_res_ptr = res_ptr;
        this->_res_id = decx::ResourceCheckIn(&this->_res_ptr, lifespan, (res_release_fn)_decon);
    }

    /*
    * @return : The raw pointer of the resource (pointer type conversion included)
    */
    template <class ResType>
    ResType* get_resource_raw_ptr() const
    {
        return static_cast<ResType*>(this->_res_ptr);
    }

    /*
    * @brief : Lock the resource so that it won't be deleted by the resource manager when using.
    */
    void lock() {
        decx::ResourceLock(this->_res_id);
    }

    /*
    * @brief : Unlock the resource to tell the resource manager to delete it when its lifespan is over.
    */
    void unlock() {
        decx::ResourceUnlock(this->_res_id);
    }
};



#endif
