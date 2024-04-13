/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "decx_resource.h"


decx::ResMgr::ResMgr()
{
    this->_run = true;
    this->_last_res_num = 0;
    this->_shortest_wait_period = 0x7fffffffffffffff;

    this->_mgr_thread = new std::thread(&decx::ResMgr::_mgr_task, this);

    this->_wp._outer_info = this;
}


void decx::ResMgr::_mgr_task()
{
    this->_shortest_wait_period = (long long)100;
    this->_last_res_num = this->_res_arr.size();

    while (this->_run)
    {
        time_t _current;
        time(&_current);

        this->_shortest_wait_period = 0x7fffffffffffffff;
        for (uint32_t i = 0; i < this->_res_arr.size(); ++i) 
        {
            decx::Resource* res_ptr = this->_res_arr[i];
            
            if (res_ptr->exceeded_lifespan(_current)) {
                this->_mtx.lock();
                if (res_ptr->Delete()) {
                    this->_res_arr.del(i);
                }
                this->_mtx.unlock();
            }
            else{
                this->_shortest_wait_period = min(this->_shortest_wait_period,
                    res_ptr->get_lifespan() - _current + res_ptr->get_last_used_instant());
            }
        }

        this->_last_res_num = this->_res_arr.size();
        {
            std::unique_lock<std::mutex> lock{ this->_mtx };
            this->_cv.wait_for(lock, std::chrono::seconds(this->_shortest_wait_period), this->_wp);
        }
    }
}


uint64_t decx::ResMgr::checkin(void** exposed_ptr, const time_t lifespan, 
    res_release_fn _decon)
{
    this->_mtx.lock();
    this->_res_arr.emplace_back(exposed_ptr, lifespan, _decon);
    this->_mtx.unlock();
    this->_cv.notify_one();
    return this->_res_arr.size() - 1;
}


void decx::ResMgr::lock_resource(const uint64_t res_id)
{
    this->_mtx.lock();
    this->_res_arr[res_id]->lock();
    this->_mtx.unlock();
}


void decx::ResMgr::unlock_resource(const uint64_t res_id)
{
    this->_mtx.lock();
    this->_res_arr[res_id]->unlock();
    this->_mtx.unlock();
}



void decx::ResMgr::checkout(const uint64_t res_id)
{
    this->_mtx.lock();
    this->_res_arr.del(res_id);
    this->_mtx.unlock();
    this->_cv.notify_one();
}


decx::ResMgr::~ResMgr()
{
    this->_mtx.lock();
    this->_run = false;
    this->_mtx.unlock();
    this->_cv.notify_one();

    if (this->_mgr_thread->joinable()) {
        this->_mgr_thread->join();
    }
    delete this->_mgr_thread;
}


_DECX_API_ uint64_t decx::ResourceCheckIn(void** exposed_ptr, const time_t lifespan_sec,
    res_release_fn _decon)
{
    return decx::_res_mgr->checkin(exposed_ptr, lifespan_sec, _decon);
}


_DECX_API_ void decx::ResourceLock(const uint64_t res_id)
{
    decx::_res_mgr->lock_resource(res_id);
}


_DECX_API_ void decx::ResourceUnlock(const uint64_t res_id)
{
    decx::_res_mgr->unlock_resource(res_id);
}


_DECX_API_ void decx::ResourceCheckOut(const uint64_t res_id)
{
    decx::_res_mgr->checkout(res_id);
}


decx::ResMgr* decx::_res_mgr;
