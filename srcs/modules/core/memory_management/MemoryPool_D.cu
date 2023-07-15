/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "MemoryPool_D.h"



decx::MemPool_D::MemPool_D()
{
    for (int i = 0; i < Init_Mem_Capacity; ++i) {
        this->mem_chunk_set_list.emplace_back(i);
    }
    this->list_length = Init_Mem_Capacity;
}



bool decx::MemPool_D::search_for_idle(size_t req_size, int begin_dex, decx::MemBlock** _ptr)
{
    bool _found = false;

    for (int i = begin_dex; i < this->list_length; ++i)
    {
        auto chunk_set = this->mem_chunk_set_list.begin() + i;
        for (int j = 0; j < chunk_set->list_length; ++j)
        {
            auto chunk = chunk_set->mem_chunk_list.begin() + j;
            for (int k = 0; k < chunk->list_length; ++k)
            {
                decx::MemBlock* _MBPtr = *(chunk->mem_block_list.begin() + k);
                if (_MBPtr->_idle) {
                    if (_MBPtr->block_size > req_size || _MBPtr->block_size == req_size) {
                        _found = true;
                        if (_MBPtr->block_size > req_size) {
                            chunk->split(k, req_size);
                        }
                        *_ptr = _MBPtr;
                        _MBPtr->_idle = false;        // lable as occupied
                        break;
                    }
                }
                if (_found) break;
            }
            if (_found) break;
        }
        if (_found) break;
    }
    return _found;
}


void decx::MemPool_D::allocate(size_t req_size, decx::MemBlock** _ptr)
{
    int begin_dex = decx::utils::_GetHighest_abd(
        decx::utils::clamp_min<size_t>(req_size, Min_Alloc_Bytes)) - dex_to_pow_bias;
    decx::MemBlock* _MBPtr = NULL;

    bool _found = this->search_for_idle(req_size, begin_dex, &_MBPtr);
    *_ptr = _MBPtr;        // assign the value to the pointer

    if (!_found)        // physically allocate one
    {
        size_t alloc_size = 1 << (begin_dex + dex_to_pow_bias);
        auto chunk_set = this->mem_chunk_set_list.begin() + begin_dex;
        chunk_set->mem_chunk_list.emplace_back(
            decx::MemChunk_D(alloc_size, req_size, begin_dex, chunk_set->list_length));

        _MBPtr = chunk_set->mem_chunk_list[chunk_set->list_length].mem_block_list[0];
        *_ptr = _MBPtr;                // assign the value to the pointer
        chunk_set->list_length++;    // increase the length of chunk_set
        _MBPtr->_ref_times = 1;        // set the reference time to one
    }
    else {
        _MBPtr->_ref_times = 1;
    }
}



void decx::MemPool_D::deallocate(decx::MemBlock* _ptr)
{
    if (_ptr != NULL) {
        decx::MemChunk_D* tmp_ptr = &this->mem_chunk_set_list[_ptr->_loc.x].mem_chunk_list[_ptr->_loc.y];
        // if the reference time is 1, which means that it will be zero when deallocated.
        // So set this block idle and check if it can be merged
        if (tmp_ptr->mem_block_list[_ptr->_loc.z]->_ref_times == 1) {
            // free the block by labeling it idle
            _ptr->_idle = true;
            tmp_ptr->check_to_merge(_ptr->_loc.z);
        }
        // otherwise, self-decrease one
        else {
            tmp_ptr->mem_block_list[_ptr->_loc.z]->_ref_times--;
        }
    }
}



void decx::MemPool_D::register_reference(decx::MemBlock* _ptr)
{
    // Increase one to the reference number
    this->mem_chunk_set_list[_ptr->_loc.x].mem_chunk_list[_ptr->_loc.y].mem_block_list[_ptr->_loc.z]->_ref_times++;
}



void decx::MemPool_D::release()
{
    for (int i = 0; i < this->list_length; ++i)
    {
        auto chunk_set = this->mem_chunk_set_list.begin() + i;
        for (int j = 0; j < chunk_set->list_length; ++j)
        {
            auto chunk = chunk_set->mem_chunk_list.begin() + j;
            if (chunk->header_ptr != NULL) {
                decx::alloc::free_D(chunk->header_ptr);
            }
        }
    }
}


decx::MemPool_D::~MemPool_D()
{
    // I can not call this->release(), cudart.dll(.so) has already exited before this
    // destructing function is called
}


decx::MemPool_D decx::mem_pool_D;