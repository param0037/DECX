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

#include "synchronize.h"
#define MODULE_TAG "Pipeline"


decx::utils::Synchronize::Synchronize() : decx::utils::NodeBase()
{
    this->_node_type = NodeTypes_e::NodeType_Synchronize;
    memset(this->_p_sync_streams, 0, MAX_CONCURRENT_BRANCHS_NUM * sizeof(std::future<void>*));
    this->_sync_streams_num = 0;
}


decx::utils::Synchronize::Synchronize(const char* node_name) : decx::utils::NodeBase(node_name)
{
    this->_node_type = NodeTypes_e::NodeType_Synchronize;
    memset(this->_p_sync_streams, 0, MAX_CONCURRENT_BRANCHS_NUM * sizeof(std::future<void>*));
    this->_sync_streams_num = 0;
}


int32_t decx::utils::Synchronize::Process()
{
    for (int32_t i = 0; i < this->_sync_streams_num; ++i){
        this->_p_sync_streams[i]->get();
    }
    return 0;
}


int32_t decx::utils::Synchronize::RegisterOneStream(decx::utils::ConcurrentSplit* p_conc_split, decx::utils::NodeBase* p_stream_head)
{
    if (p_conc_split == nullptr){
        DECX_LOG_ERR("%s failed to register stream since the stream header is NULL", this->_name);
        return -1;
    }
    auto* info = p_conc_split->GetStreamInfoByStreamHeader(p_stream_head);
    if (info == nullptr){
        DECX_LOG_ERR("%s failed to register stream since the stream header is not found in the list", this->_name);
        return -1;
    }
    this->_p_sync_streams[this->_sync_streams_num] = &info->_future;
    this->_sync_streams_num++;
    return 0;
}