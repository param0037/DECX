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

#ifndef _NODE_BASE_H_
#define _NODE_BASE_H_

#include <basic.h>
#include <log_console.h>
#include <PtrInfo.h>


namespace decx
{
namespace utils
{
    class Pipeline;

    class NodeBase;


    typedef int32_t NodeTaskFunc_t(const void**, void**, void**);
    typedef int32_t PredicatorFunc_t(const void*, int32_t*);


    enum class NodeTypes_e
    {
        NodeType_Base               = 0,
        NodeType_Task               = 1,
        NodeType_BranchSplit        = 2,
        NodeType_ConcurrentSplit    = 3,
        NodeType_BranchMerge        = 4,
        NodeType_Synchronize        = 5,
    };
}
}

#define NODE_NAME_MAX_LENGTH 32
#define NODE_DATA_BUFFER_SIZE 1024


class decx::utils::NodeBase
{
friend class decx::utils::Pipeline;

protected:
    char        _name[NODE_NAME_MAX_LENGTH];
    uint32_t    _thread_id;
    void*       _task_func;

    decx::utils::NodeBase* _prev;
    decx::utils::NodeBase* _next;

    NodeTypes_e _node_type;

public:
    NodeBase();


    NodeBase(const char* node_name);


    NodeTypes_e GetNodeType() const;


    const char* GetNodeName() const;


    int32_t SetUpStreamNode(decx::utils::NodeBase* p_prev);
    int32_t SetDownStreamNode(decx::utils::NodeBase* p_next);


    decx::utils::NodeBase* GetDownStreamNode();
    decx::utils::NodeBase* GetUpStreamNode();


    virtual int32_t Process();


    virtual ~NodeBase();
};

#endif
