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

#ifndef _CONCURRENT_SPLIT_H_
#define _CONCURRENT_SPLIT_H_

#include "node_base.h"
#include <thread_management/thread_pool.h>

namespace decx
{
namespace utils
{
    class ConcurrentSplit;
    class Synchronize;

    enum class StreamThreadDispatchMethod_e
    {
        Dispatch_NewSlot = 0,
        Dispatch_LoadBalanced = 1,
        Dispatch_ByID = 2,
    };

    struct StreamHeaderInfo_t
    {
        decx::utils::NodeBase*       _p_stream_head;
        int32_t                      _thread_id;
        decx::utils::Synchronize*    _p_sync;
        std::future<void>            _future;
        StreamThreadDispatchMethod_e _dispatch_method;
    };
}
}

#define MAX_CONCURRENT_BRANCHS_NUM 64

class decx::utils::ConcurrentSplit : public decx::utils::NodeBase
{
private:
    decx::utils::StreamHeaderInfo_t _header_info_arr[MAX_CONCURRENT_BRANCHS_NUM];
    uint32_t _branch_num;

    static _THREAD_FUNCTION_ void BranchFunctionByID(decx::utils::ConcurrentSplit* _fake_this, const uint32_t branch_id);

public:
    ConcurrentSplit();


    ConcurrentSplit(const char* node_name);


    decx::utils::StreamHeaderInfo_t* GetStreamInfoByID(const uint32_t id);
    decx::utils::StreamHeaderInfo_t* GetStreamInfoByStreamHeader(const decx::utils::NodeBase* p_stream_header);


    int32_t RegisterBranchHead(decx::utils::NodeBase* p_conc_split, const StreamThreadDispatchMethod_e method = StreamThreadDispatchMethod_e::Dispatch_NewSlot,
        const int32_t slot_id = 0);


    virtual int32_t Process() override;
};

#endif