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
