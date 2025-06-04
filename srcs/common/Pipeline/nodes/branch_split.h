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

#ifndef _BRANCH_SPLIT_H_
#define _BRANCH_SPLIT_H_

#include "node_base.h"


namespace decx
{
namespace utils
{
    class BranchSplit;
}
}


#define MAX_BRANCH_NUM 32
#define PREDICATED_DATA_MAX_LENGTH 1024


class decx::utils::BranchSplit : public decx::utils::NodeBase
{
private:
    decx::PtrInfo<void> _node_data_buf;
    decx::utils::NodeBase* _branch_heads[MAX_BRANCH_NUM];
    uint32_t _branch_num;

    const void* _data_in;

public:
    BranchSplit();


    BranchSplit(const char* node_name);


    int32_t SetPredicatedData(void* p_data, const uint64_t size, const bool use_buitin_buffer = false, de::DH* handle = nullptr);


    int32_t RegisterBranchHead(decx::utils::NodeBase* p_branch_head);


    int32_t PredicatorRegister(PredicatorFunc_t* node_func);


    int32_t AllocateNodeBufData(de::DH* handle);


    virtual int32_t Process() override;
};

#endif