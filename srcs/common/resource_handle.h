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

#ifndef _RESOURCE_HANDLE_H_
#define _RESOURCE_HANDLE_H_

#include <basic.h>

namespace decx
{
    struct ResourceHandle;

    typedef void (*res_release_fn)(void*);


    _DECX_API_ uint64_t ResourceCheckIn(void** exposed_ptr, const time_t lifespan_sec, res_release_fn _decon);


    _DECX_API_ void ResourceLock(const uint64_t res_id);


    _DECX_API_ void ResourceUnlock(const uint64_t res_id);


    _DECX_API_ void ResourceCheckOut(const uint64_t res_id);
}


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
    * @param destructor_callback :   The callback function when checkout the resource. 
    *                   Note : This callback function must be in type void func(type*).
    *                   If the function is a member function of a class, plase define it as static.
    */
    template <class _decon_type>
    void RegisterResource(void* res_ptr, const time_t lifespan, _decon_type* destructor_callback = NULL)
    {
        // Prevent duplicate registration to the same resource
        if (this->_res_ptr == NULL)
        {
            this->_res_ptr = res_ptr;
            this->_res_id = decx::ResourceCheckIn(&this->_res_ptr, lifespan, (res_release_fn)destructor_callback);
        }
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
