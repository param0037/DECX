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

#ifndef _PIPELINE_H_
#define _PIPELINE_H_

#include <basic.h>
#include "nodes/node_base.h"
#include "nodes/branch_split.h"
#include <Array/Dynamic_Array.h>
#include <thread_management/thread_arrange.h>

#define PIPELINE_THREAD_ID_START 0

namespace decx
{
    namespace utils{
        class Pipeline;
    }
}


class decx::utils::Pipeline
{
private:
    decx::utils::Dynamic_Array<decx::utils::NodeBase*> _node_ptr_arr;
    decx::utils::Thr1D _thread_handlers;

    
    int32_t NodeFinder(const decx::utils::NodeBase* target, decx::utils::NodeBase** p_res);

public:
    Pipeline();


    int32_t Link(std::initializer_list<decx::utils::NodeBase*> node_ptrs);


    int32_t AddBranch(decx::utils::BranchSplit* branch_split, std::initializer_list<decx::utils::NodeBase*> branch);


    int32_t Run();
};

#endif
