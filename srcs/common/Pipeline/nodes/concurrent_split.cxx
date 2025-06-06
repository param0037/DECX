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

#include "concurrent_split.h"
#define MODULE_TAG "Pipeline"


decx::utils::ConcurrentSplit::ConcurrentSplit() : decx::utils::NodeBase()
{
    memset(this->_header_info_arr, 0, sizeof(decx::utils::StreamHeaderInfo_t) * MAX_CONCURRENT_BRANCHS_NUM);
    this->_node_type = NodeTypes_e::NodeType_ConcurrentSplit;
    this->_branch_num = 0;
}


decx::utils::ConcurrentSplit::ConcurrentSplit(const char* node_name) : decx::utils::NodeBase(node_name)
{
    memset(this->_header_info_arr, 0, sizeof(decx::utils::StreamHeaderInfo_t) * MAX_CONCURRENT_BRANCHS_NUM);
    this->_node_type = NodeTypes_e::NodeType_ConcurrentSplit;
    this->_branch_num = 0;
}


int32_t decx::utils::ConcurrentSplit::
RegisterBranchHead(decx::utils::NodeBase* p_branch_head,
                   const StreamThreadDispatchMethod_e method,
                   const int32_t slot_id)
{
    if (p_branch_head == nullptr){
        DECX_LOG_ERR("%s: failed to register branch header, since its pointer is NULL", this->_name);
        return -1;
    }
    if (this->_branch_num > MAX_CONCURRENT_BRANCHS_NUM - 1){
        DECX_LOG_ERR("%s: failed to register branch header, since quantity limit is already exceeded", this->_name);
        return -1;
    }
    auto& info = this->_header_info_arr[this->_branch_num];
    info._p_stream_head = p_branch_head;
    info._dispatch_method = method;
    info._thread_id = slot_id;
    this->_branch_num++;
    return 0;
}


decx::utils::StreamHeaderInfo_t* decx::utils::ConcurrentSplit::GetStreamInfoByID(const uint32_t id)
{
    return this->_header_info_arr + id;
}


decx::utils::StreamHeaderInfo_t* 
decx::utils::ConcurrentSplit::GetStreamInfoByStreamHeader(const decx::utils::NodeBase* p_stream_header)
{
    for (int32_t i = 0; i < this->_branch_num; ++i){
        uint64_t addr_head_in_list = (uint64_t)(this->_header_info_arr[i]._p_stream_head);
        if (addr_head_in_list == (uint64_t)p_stream_header){
            return this->_header_info_arr + i;
        }
    }
    return nullptr;
}


_THREAD_FUNCTION_ void
decx::utils::ConcurrentSplit::BranchFunctionByID(decx::utils::ConcurrentSplit* _fake_this, const uint32_t branch_id)
{
    auto* p_branch_head = _fake_this->_header_info_arr[branch_id]._p_stream_head;
    decx::utils::NodeBase* p_branch_node = p_branch_head;
    while (p_branch_node->GetNodeType() != NodeTypes_e::NodeType_Synchronize)
    {
        p_branch_node->Process();
        p_branch_node = p_branch_node->GetNextNodeBasePtr();
        if (p_branch_node == nullptr)
            break;
    }
}


int32_t decx::utils::ConcurrentSplit::Process()
{
    for (int32_t i = 0; i < this->_branch_num; ++i)
    {
        auto& info = this->_header_info_arr[i];
        switch (this->_header_info_arr[i]._dispatch_method)
        {
        case StreamThreadDispatchMethod_e::Dispatch_NewSlot:
            info._future = decx::cpu::RegisterTaskAppened(BranchFunctionByID, this, i);
            break;

        case StreamThreadDispatchMethod_e::Dispatch_LoadBalanced:
            info._future = decx::cpu::RegisterTaskLoadBalanced(BranchFunctionByID, this, i);
            break;

        case StreamThreadDispatchMethod_e::Dispatch_ByID:
            info._future = decx::cpu::RegisterTaskByID(BranchFunctionByID, info._thread_id, this, i);
            break;
        
        default:
            break;
        }
        
    }
    return 0;
}
