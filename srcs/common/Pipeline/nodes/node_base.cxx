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

#include "node_base.h"
#include <string.h>
#define MODULE_TAG "Pipeline"


decx::utils::NodeBase::NodeBase()
{
    memset(this->_name, 0, NODE_NAME_MAX_LENGTH);
    this->_prev = nullptr;
    this->_next = nullptr;

    this->_node_type = NodeTypes_e::NodeType_Base;
}


decx::utils::NodeBase::NodeBase(const char* node_name)
{
    this->_prev = nullptr;
    this->_next = nullptr;
    memset(this->_name, 0, NODE_NAME_MAX_LENGTH);
    strcpy(this->_name, node_name);
    this->_node_type = NodeTypes_e::NodeType_Base;
}


int32_t decx::utils::NodeBase::Process()
{
    return 0;
}


decx::utils::NodeBase::~NodeBase()
{
    return;
}


int32_t decx::utils::NodeBase::SetUpStreamNode(decx::utils::NodeBase* p_prev)
{
    this->_prev = p_prev;
    return 0;
}


const char* decx::utils::NodeBase::GetNodeName() const
{
    return this->_name;
}


decx::utils::NodeTypes_e decx::utils::NodeBase::GetNodeType() const
{
    return this->_node_type;
}


int32_t decx::utils::NodeBase::SetDownStreamNode(decx::utils::NodeBase* p_next)
{
    this->_next = p_next;
    return 0;
}