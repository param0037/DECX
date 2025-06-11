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

#include "branch_split.h"
#define MODULE_TAG "Pipeline"


decx::utils::BranchSplit::BranchSplit() : decx::utils::NodeBase()
{
    memset(this->_branch_heads, 0, MAX_BRANCH_NUM * sizeof(decx::utils::NodeBase*));
    this->_branch_num = 0;
    this->_node_type = NodeTypes_e::NodeType_BranchSplit;
}


decx::utils::BranchSplit::BranchSplit(const char* node_name) : decx::utils::NodeBase(node_name)
{
    memset(this->_branch_heads, 0, MAX_BRANCH_NUM * sizeof(decx::utils::NodeBase*));
    this->_branch_num = 0;
    this->_node_type = NodeTypes_e::NodeType_BranchSplit;
}


int32_t decx::utils::BranchSplit::AllocateNodeBufData(de::DH* handle)
{
    if (this->_node_data_buf.IsValid() == 0)
        return this->_node_data_buf.Allocate(NODE_DATA_BUFFER_SIZE, PAGABLE, handle);

    return 0;
}

int32_t decx::utils::BranchSplit::SetPredicatedData(void* p_data, const uint64_t size, const bool use_buitin_buffer, de::DH* handle)
{
    if (use_buitin_buffer) {
        this->AllocateNodeBufData(handle);
        if (p_data == nullptr){
            DECX_LOG_ERR("%s: failed to set predicated data since its pointer is NULL", this->_name);
            return -1;
        }
        if (size > PREDICATED_DATA_MAX_LENGTH){
            DECX_LOG_ERR("%s: failed to set predicated data since it is oversized", this->_name);
            return -1;
        }
        this->_data_in = (void*)this->_node_data_buf;
        memcpy((void*)this->_node_data_buf, p_data, size);
    }
    else{
        this->_data_in = p_data;
    }
    return 0;
}


int32_t decx::utils::BranchSplit::Process()
{
    if (this->_task_func == nullptr){
        DECX_LOG_ERR("%s failed to run a null predicator");
        return -1;
    }
    auto* p_task_func = (PredicatorFunc_t*)this->_task_func;
    int32_t slot_idx;
    int32_t rval = (*p_task_func)((const void*)this->_data_in, &slot_idx);
    if (slot_idx >= this->_branch_num){
        DECX_LOG_ERR("Jump rejected, since the index is out of range");
        return -1;
    }
    this->_next = _branch_heads[slot_idx];
    return rval;
}


int32_t decx::utils::BranchSplit::RegisterBranchHead(decx::utils::NodeBase* p_branch_head)
{
    if (this->_branch_num > MAX_BRANCH_NUM - 1){
        DECX_LOG_ERR("Failed to register branch header, since quantity limit is already exceeded");
        return -1;
    }
    this->_branch_heads[this->_branch_num] = p_branch_head;
    this->_branch_num++;
    return 0;
}


int32_t decx::utils::BranchSplit::PredicatorRegister(PredicatorFunc_t* node_func)
{
    if (node_func == nullptr){
        return -1;
    }
    this->_task_func = (void*)node_func;
    return 0;
}