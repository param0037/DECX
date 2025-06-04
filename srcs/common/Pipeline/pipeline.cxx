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

#include "pipeline.h"
#define MODULE_TAG "Pipeline"

decx::utils::Pipeline::Pipeline()
{
    memset(this, 0, sizeof(decx::utils::Pipeline));
    this->_node_ptr_arr.Init();
}


int32_t decx::utils::Pipeline::Link(std::initializer_list<decx::utils::NodeBase*> nodes)
{
    int32_t idx = 0;
    int32_t rval = 0;
    for (auto p_node = nodes.begin(); p_node != nodes.end(); ++p_node){
        this->_node_ptr_arr.emplace_back(*p_node);

        decx::utils::NodeBase* p_next_node = nullptr;
        if (idx < nodes.size() - 1){
            p_next_node = *(p_node + 1);
        }
        decx::utils::NodeBase* p_prev_node = nullptr;
        if (idx > 0){
            p_prev_node = *(p_node - 1);
        }
        rval |= (*p_node)->SetUpStreamNode(p_prev_node);
        rval |= (*p_node)->SetDownStreamNode(p_next_node);
        ++idx;
    }
    return rval;
}


int32_t decx::utils::Pipeline::NodeFinder(const decx::utils::NodeBase* target, decx::utils::NodeBase** p_res)
{
    for (int32_t i = 0; i < this->_node_ptr_arr.size(); ++i){
        if (this->_node_ptr_arr[i] == target){
            if (p_res != nullptr)
                *p_res = this->_node_ptr_arr[i];
            return 1;
        }
    }
    if (p_res != nullptr)
        *p_res = nullptr;
    return 0;
}


int32_t decx::utils::Pipeline::Run()
{
    int32_t rval = 0;

    decx::utils::NodeBase* p_node = this->_node_ptr_arr[0];
    while (p_node != nullptr)
    {
        rval |= p_node->Process();
        p_node = p_node->_next;
    }
    return rval;
}


int32_t decx::utils::Pipeline::AddBranch(decx::utils::BranchSplit* branch_split, std::initializer_list<decx::utils::NodeBase*> branch)
{
    if (branch_split == nullptr){
        DECX_LOG_ERR("Failed to add branch, since the branch header is NULL");
        return -1;
    }
    if (branch.size() == 0){
        return 0;
    }

    int32_t rval = 0;
    auto* _branch_starter = branch.begin();
    rval |= branch_split->SetUpStreamNode(nullptr);
    rval |= branch_split->SetDownStreamNode(*_branch_starter);
    rval |= (*_branch_starter)->SetUpStreamNode(branch_split);

    rval |= branch_split->RegisterBranchHead(*_branch_starter);

    // if (this->NodeFinder(branch_split, nullptr) == 0){
    //     this->_node_ptr_arr.emplace_back(branch_split);
    // }

    int32_t idx = 0;
    for (auto* p_branch_node = branch.begin(); p_branch_node != branch.end(); ++p_branch_node)
    {
        decx::utils::NodeBase* p_next_node = nullptr;
        if (idx < branch.size() - 1){
            p_next_node = *(p_branch_node + 1);
        }
        decx::utils::NodeBase* p_prev_node = nullptr;
        if (idx > 0){
            p_prev_node = *(p_branch_node - 1);
        }

        (*p_branch_node)->SetUpStreamNode(p_prev_node);
        (*p_branch_node)->SetDownStreamNode(p_next_node);
        // Emplace back the node pointer
        this->_node_ptr_arr.emplace_back(*p_branch_node);
    }
    return rval;
}