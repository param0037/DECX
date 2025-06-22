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
    memset(&_data_in, 0, sizeof(TaskNode_Stack_t<void*, NODE_DATA_BUFFER_SIZE>));
    memset(&_data_exchanged, 0, sizeof(TaskNode_Stack_t<void*, NODE_DATA_BUFFER_SIZE>));
    memset(&_data_out, 0, sizeof(TaskNode_Stack_t<void*, NODE_DATA_BUFFER_SIZE>));
}


decx::utils::TaskNode::TaskNode(const char* node_name) : decx::utils::NodeBase(node_name)
{
    this->_node_type = NodeTypes_e::NodeType_Task;
    memset(&_data_in, 0, sizeof(TaskNode_Stack_t<void*, NODE_DATA_BUFFER_SIZE>));
    memset(&_data_exchanged, 0, sizeof(TaskNode_Stack_t<void*, NODE_DATA_BUFFER_SIZE>));
    memset(&_data_out, 0, sizeof(TaskNode_Stack_t<void*, NODE_DATA_BUFFER_SIZE>));
}


int32_t decx::utils::TaskNode::NodeTaskRegister(decx::utils::NodeTaskFunc_t* node_func)
{
    if (node_func == nullptr){
        return -1;
    }
    this->_task_func = (void*)node_func;
    return 0;
}


int32_t decx::utils::TaskNode::AllocateNodeBufData(de::DH* handle, TaskNode_WorkingData_Type_e type)
{
    switch (type)
    {
    case TaskNode_WorkingData_Type_e::TaskNode_Data_ReadOnly:
    if (this->_data_in._aux_buffer.IsValid() == 0)
        return this->_data_in._aux_buffer.Allocate(NODE_DATA_BUFFER_SIZE, PAGABLE, handle);

    case TaskNode_WorkingData_Type_e::TaskNode_Data_Swap:
    if (this->_data_exchanged._aux_buffer.IsValid() == 0)
        return this->_data_exchanged._aux_buffer.Allocate(NODE_DATA_BUFFER_SIZE, PAGABLE, handle);
    
    case TaskNode_WorkingData_Type_e::TaskNode_Data_Write:
    if (this->_data_out._aux_buffer.IsValid() == 0)
        return this->_data_out._aux_buffer.Allocate(NODE_DATA_BUFFER_SIZE, PAGABLE, handle);

    default:
        break;
    }

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
    return (*p_task_func)((const void**)this->_data_in._checkpoints, (void**)this->_data_exchanged._checkpoints, 
        (void**)this->_data_out._checkpoints);
}


int32_t decx::utils::TaskNode::SetData(TaskNode_WorkingData_Type_e  type, 
                                       const void*                  p_data, 
                                       const uint64_t               size, 
                                       const bool                   use_builtin_buf, 
                                       de::DH*                      handle)
{
    if (use_builtin_buf) {
        this->AllocateNodeBufData(handle, type);
        if (p_data == nullptr){
            DECX_LOG_ERR("Failed to set input data since the pointer is NULL");
            return -1;
        }
        if (size > NODE_NAME_MAX_LENGTH){
            DECX_LOG_ERR("Failed to set input data since it is oversized");
            return -1;
        }
    }

    switch (type)
    {
    case TaskNode_WorkingData_Type_e::TaskNode_Data_ReadOnly:
        if (use_builtin_buf) {
            uint8_t* p_checkpoint = (uint8_t*)this->_data_exchanged._aux_buffer + this->_data_exchanged._stack_aux_buffer_head;
            this->_data_in._checkpoints[this->_data_in._current_checkpoint_num] = p_checkpoint;
            memcpy(p_checkpoint, p_data, size);
            this->_data_in._stack_aux_buffer_head += size;
        }
        else{
            this->_data_in._checkpoints[this->_data_in._current_checkpoint_num] = const_cast<void*>(p_data);
        }
        this->_data_in._current_checkpoint_num++;
        break;

    case TaskNode_WorkingData_Type_e::TaskNode_Data_Swap:
        if (use_builtin_buf) {
            uint8_t* p_checkpoint = (uint8_t*)this->_data_exchanged._aux_buffer + this->_data_exchanged._stack_aux_buffer_head;
            this->_data_exchanged._checkpoints[this->_data_exchanged._current_checkpoint_num] = p_checkpoint;
            memcpy(p_checkpoint, p_data, size);
            this->_data_exchanged._stack_aux_buffer_head += size;
        }
        else{
            this->_data_exchanged._checkpoints[this->_data_exchanged._current_checkpoint_num] = const_cast<void*>(p_data);
        }
        this->_data_exchanged._current_checkpoint_num++;
        break;
    
    case TaskNode_WorkingData_Type_e::TaskNode_Data_Write:
        if (use_builtin_buf) {
            uint8_t* p_checkpoint = (uint8_t*)this->_data_exchanged._aux_buffer + this->_data_exchanged._stack_aux_buffer_head;
            this->_data_out._checkpoints[this->_data_out._current_checkpoint_num] = p_checkpoint;
            memcpy(p_checkpoint, p_data, size);
            this->_data_out._stack_aux_buffer_head += size;
        }
        else{
            this->_data_out._checkpoints[this->_data_out._current_checkpoint_num] = const_cast<void*>(p_data);
        }
        this->_data_out._current_checkpoint_num++;
        break;

    default:
        break;
    }
    return 0;
}


decx::utils::TaskNode::~TaskNode()
{
    if (this->_data_in._aux_buffer.IsValid()){
        this->_data_in._aux_buffer.Free();
    }
    if (this->_data_exchanged._aux_buffer.IsValid()){
        this->_data_exchanged._aux_buffer.Free();
    }
    if (this->_data_out._aux_buffer.IsValid()){
        this->_data_out._aux_buffer.Free();
    }
}
