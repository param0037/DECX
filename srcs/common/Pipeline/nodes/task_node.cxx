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

#include "task_node.h"
#define MODULE_TAG "Pipeline"


decx::utils::TaskNode::TaskNode() : decx::utils::NodeBase()
{
    this->_node_type = NodeTypes_e::NodeType_Task;
    this->_data_in = nullptr;
    this->_data_out = nullptr;
}


decx::utils::TaskNode::TaskNode(const char* node_name) : decx::utils::NodeBase(node_name)
{
    this->_node_type = NodeTypes_e::NodeType_Task;
    this->_data_in = nullptr;
    this->_data_out = nullptr;
}


int32_t decx::utils::TaskNode::NodeTaskRegister(decx::utils::NodeTaskFunc_t* node_func)
{
    if (node_func == nullptr){
        return -1;
    }
    this->_task_func = (void*)node_func;
    return 0;
}


int32_t decx::utils::TaskNode::AllocateNodeBufData(de::DH* handle)
{
    if (this->_node_data_buf.IsValid() == 0)
        return this->_node_data_buf.Allocate(NODE_DATA_BUFFER_SIZE * 3, PAGABLE, handle);

    return 0;
}


int32_t decx::utils::TaskNode::Process()
{
    if (this->_task_func == nullptr){
        return -1;
    }
    auto* p_task_func = (NodeTaskFunc_t*)this->_task_func;
    if (nullptr == p_task_func){
        DECX_LOG_ERR("%s, failed to run node task, since the function pointer is NULL");
        return -1;
    }
    return (*p_task_func)((const void*)this->_data_in, (void*)this->_data_exchanged, (void*)this->_data_out);
}


int32_t decx::utils::TaskNode::SetData(TaskNode_WorkingData_Type_e  type, 
                                       const void*                  p_data, 
                                       const uint64_t               size, 
                                       const bool                   use_builtin_buf, 
                                       de::DH*                      handle)
{
    if (use_builtin_buf) {
        this->AllocateNodeBufData(handle);
        if (p_data == nullptr){
            DECX_LOG_ERR("Failed to set input data since the pointer is NULL");
            return -1;
        }
        if (size > NODE_NAME_MAX_LENGTH){
            DECX_LOG_ERR("Failed to set input data since it is oversized");
            return -1;
        }
        this->_data_in = (void*)this->_node_data_buf;
        memcpy((void*)this->_node_data_buf, p_data, size);
    }
    else{
        this->_data_in = p_data;
    }

    switch (type)
    {
    case TaskNode_WorkingData_Type_e::TaskNode_Data_ReadOnly:
        if (use_builtin_buf){
            this->_data_in = (void*)this->_node_data_buf;
            memcpy((void*)this->_node_data_buf, p_data, size);
        }
        else{
            this->_data_in = p_data;
        }
        break;

    case TaskNode_WorkingData_Type_e::TaskNode_Data_Swap:
        if (use_builtin_buf){
            this->_data_exchanged = (uint8_t*)this->_node_data_buf + NODE_DATA_BUFFER_SIZE;
            memcpy((uint8_t*)this->_node_data_buf + NODE_DATA_BUFFER_SIZE, p_data, size);
        }
        else{
            this->_data_exchanged = const_cast<void*>(p_data);
        }
        break;
    
    case TaskNode_WorkingData_Type_e::TaskNode_Data_Write:
        if (use_builtin_buf){
            this->_data_out = (uint8_t*)this->_node_data_buf + NODE_DATA_BUFFER_SIZE * 2;
            memcpy((uint8_t*)this->_node_data_buf + NODE_DATA_BUFFER_SIZE * 2, p_data, size);
        }
        else{
            this->_data_out = const_cast<void*>(p_data);
        }
        break;

    default:
        break;
    }
    return 0;
}


int32_t decx::utils::TaskNode::AssignOutBufPtr(void* p_out_buf)
{
    if (p_out_buf == nullptr){
        DECX_LOG_ERR("Failed to assign output buffer since the pointer is NULL");
        return -1;
    }
    this->_data_out = (uint8_t*)p_out_buf;
    return 0;
}


decx::utils::TaskNode::~TaskNode()
{
    if (this->_node_data_buf.IsValid()){
        this->_node_data_buf.Free();
    }
}
