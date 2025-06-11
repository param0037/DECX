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

#ifndef _TASK_IO_NODE_H_
#define _TASK_IO_NODE_H_


#include "node_base.h"
#include <PtrInfo.h>


namespace decx
{
namespace utils
{
    class TaskNode;

    enum class TaskNode_WorkingData_Type_e
    {
        TaskNode_Data_ReadOnly = 0,
        TaskNode_Data_Swap = 1,
        TaskNode_Data_Write = 2,
        TaskNode_Data_TypesNum
    };


    template <typename _data_type, uint32_t MaxCPNum>
    struct TaskNode_Stack_t
    {
        decx::PtrInfo<_data_type> _aux_buffer;
        _data_type* _checkpoints[MaxCPNum];
        uint32_t _stack_aux_buffer_head;
        uint32_t _current_checkpoint_num;
    };
}
}


class decx::utils::TaskNode : public decx::utils::NodeBase
{
private:
    TaskNode_Stack_t<void, NODE_DATA_BUFFER_SIZE> _data_in;
    TaskNode_Stack_t<void, NODE_DATA_BUFFER_SIZE> _data_exchanged;
    TaskNode_Stack_t<void, NODE_DATA_BUFFER_SIZE> _data_out;

public:
    TaskNode();


    TaskNode(const char* node_name);

    
    int32_t NodeTaskRegister(NodeTaskFunc_t* node_func);


    int32_t AllocateNodeBufData(de::DH* handle, TaskNode_WorkingData_Type_e type);


    int32_t SetData(TaskNode_WorkingData_Type_e type, const void* p_data, const uint64_t size, 
        const bool use_builtin_buf = false, de::DH* handle = nullptr);


    virtual int32_t Process() override;


    virtual ~TaskNode() override;
};

#endif