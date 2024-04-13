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


decx::Resource::Resource()
{
    memset(this, 0, sizeof(decx::Resource));
}


decx::Resource::Resource(void** exposed_ptr, const time_t lifespan_sec, res_release_fn rel_fn)
{
    this->_deconstructor_callback = rel_fn;
    this->_occupied = false;

    this->_exposed_ptr = exposed_ptr;
    this->_lifespan_sec = lifespan_sec;
    time(&this->_last_used_instant);
}


bool decx::Resource::Delete()
{
    if (!this->_occupied) {
        if (*(this->_deconstructor_callback)) {     // Not NULL
            (*(this->_deconstructor_callback))(*this->_exposed_ptr);
        }
        delete* (this->_exposed_ptr);
        *(this->_exposed_ptr) = NULL;
    }
    return !this->_occupied;
}


bool decx::Resource::exceeded_lifespan(const time_t now) const
{
    return now - this->_last_used_instant + 1 > this->_lifespan_sec;
}


time_t decx::Resource::get_last_used_instant() const
{
    return this->_last_used_instant;
}


time_t decx::Resource::get_lifespan() const
{
    return this->_lifespan_sec;
}


void decx::Resource::lock()
{
    this->_occupied = true;
}


void decx::Resource::unlock()
{
    this->_occupied = false;
    time(&this->_last_used_instant);
}
