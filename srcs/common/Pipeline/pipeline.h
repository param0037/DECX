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
#include "nodes/concurrent_split.h"
#include "nodes/synchronize.h"
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

    /**
     * @brief Link nodes to pipeline main branch, launched by main thread of pipeline.alignas
     * @param node_ptrs List of the nodes (they can also be branch split or concurrent split, etc as well).
     * @return 0 for no error; Non-zero for error occuring.
     */
    int32_t LinkNodes(std::initializer_list<decx::utils::NodeBase*> node_ptrs);

    /**
     * @brief Link a branch for pipeline, this branch is also executed by the main thread, acting as conditional jump instructions.
     * @param branch_split Pointer of branch split node, indicating where this branch is attached to.
     * @param branch A list of branch nodes (they can also be branch split or concurrent split, etc as well).
     * @return 0 for no error; Non-zero for error occuring.
     */
    int32_t LinkBranch(decx::utils::BranchSplit* branch_split, std::initializer_list<decx::utils::NodeBase*> branch);

    /**
     * @brief Link a concurrent stream to pipeline
     * @param conc_split Pointer of concurrent split node, indicating where the stream is launched. If this node is linked by
     *     Pipeline::LinkNodes(), pipeline main thread will launch this stream.
     * @param stream_nodes A list of stream task nodes (they can also be branch split or concurrent split, etc as well).
     * @param backend_sync Pointer of synchronize node, indicating where the stream ends. If this node is linked by Pipeline::LinkNodes(),
     *     this stream will be synchronized with pipeline main thread, thus creating a barrier to the main thread.
     * @return 0 for no error; Non-zero for error occuring.
     */
    int32_t LinkStream(decx::utils::ConcurrentSplit* conc_split, std::initializer_list<decx::utils::NodeBase*> stream_nodes, 
        decx::utils::Synchronize* backend_sync);


    int32_t Run();
};

#endif
