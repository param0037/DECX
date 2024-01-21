/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "MemChunk_D.h"


decx::MemChunk_D::MemChunk_D(int pool_dex, int chunk_set_dex)
{
    this->chunk_size = 0;
    this->header_ptr = NULL;

    decx::MemBlock* new_node = (decx::MemBlock*)malloc(sizeof(decx::MemBlock));
    new_node->block_size = 0;
    new_node->_idle = true;
    new_node->_loc.x = pool_dex;
    new_node->_loc.y = chunk_set_dex;
    new_node->_loc.z = 0;
    new_node->_ptr = NULL;
    new_node->_prev = NULL;
    new_node->_next = NULL;
    new_node->_ref_times = 0;

    this->mem_block_list.emplace_back(new_node);

    this->list_length = this->mem_block_list.size();
}



decx::MemChunk_D::MemChunk_D(size_t size, size_t req_size, int pool_dex, int chunk_set_dex)
{
    this->chunk_size = size;

    uchar* ptr = (uchar*)decx::alloc::malloc_D(size, host_mem_alignment);
    this->header_ptr = ptr;
    
    decx::MemBlock* new_node_0 = (decx::MemBlock*)malloc(sizeof(decx::MemBlock));
    new_node_0->block_size = req_size;
    new_node_0->_idle = false;
    new_node_0->_loc.x = pool_dex;
    new_node_0->_loc.y = chunk_set_dex;
    new_node_0->_loc.z = 0;
    new_node_0->_ptr = ptr;
    new_node_0->_prev = NULL;
    new_node_0->_next = NULL;
    new_node_0->_ref_times = 0;

    this->mem_block_list.emplace_back(new_node_0);

    if (size != req_size) 
    {
        decx::MemBlock* new_node_1 = (decx::MemBlock*)malloc(sizeof(decx::MemBlock));
        new_node_1->block_size = size - req_size;
        new_node_1->_idle = true;
        new_node_1->_loc.x = pool_dex;
        new_node_1->_loc.y = chunk_set_dex;
        new_node_1->_loc.z = 1;
        new_node_1->_ptr = ptr + req_size;
        new_node_1->_next = NULL;
        new_node_1->_ref_times = 0;

        new_node_0->_next = new_node_1;
        new_node_1->_prev = new_node_0;
        this->mem_block_list.emplace_back(new_node_1);
    }
    this->list_length = this->mem_block_list.size();
}



decx::MemBlock* decx::MemChunk_D::split(int32_t dex, uint64_t req_size)
{
    decx::MemBlock* block_split = *(this->mem_block_list.begin() + dex);
    
    uint64_t splited_size = block_split->block_size - req_size;
    
    decx::MemBlock* block_insert = (decx::MemBlock*)malloc(sizeof(decx::MemBlock));
    block_insert->block_size = splited_size;
    block_insert->_idle = true;
    block_insert->_loc = block_split->_loc;
    block_insert->_loc.z = block_split->_loc.z + 1;
    block_insert->_ptr = block_split->_ptr + req_size;
    block_insert->_prev = block_split;
    block_insert->_next = NULL;
    block_insert->_ref_times = 0;

    block_split->block_size = req_size;

    if (block_split->_next != NULL)
    {                                    // not the last block
        block_insert->_next = block_split->_next;
        for (int i = dex + 1; i < this->list_length; ++i) {
            this->mem_block_list[i]->_loc.z++;
        }
        this->mem_block_list.insert(this->mem_block_list.begin() + dex + 1, block_insert);
    }
    else {        // is the last block
        this->mem_block_list.emplace_back(block_insert);
    }

    block_split->_next = block_insert;
    return block_insert;
}



decx::MemBlock* decx::MemChunk_D::forward_merge2(int dex)
{
    decx::MemBlock* current_bl = *(this->mem_block_list.begin() + dex);
    decx::MemBlock* prev_bl = current_bl->_prev;
    // capacity merged
    prev_bl->block_size += current_bl->block_size;
    if (current_bl->_next != NULL) {        // not the last one
        decx::MemBlock* next_bl = current_bl->_next;
        next_bl->_prev = prev_bl;
        prev_bl->_next = next_bl;

        for (int i = dex + 1; i < this->list_length; ++i) {
            this->mem_block_list[i]->_loc.z--;
        }

        this->mem_block_list.erase(this->mem_block_list.begin() + dex);
        free(current_bl);
    }
    else {    // is the last block
        prev_bl->_next = NULL;
        this->mem_block_list.pop_back();
        free(current_bl);
    }

    return prev_bl;
}


decx::MemBlock* decx::MemChunk_D::backward_merge2(int dex)
{
    return decx::MemChunk_D::forward_merge2(dex + 1);
}


decx::MemBlock* decx::MemChunk_D::merge3(int dex)
{
    decx::MemBlock* this_bl = *(this->mem_block_list.begin() + dex);
    decx::MemBlock* prev_bl = this_bl->_prev;
    decx::MemBlock* next_bl = this_bl->_next;

    prev_bl->block_size += (this_bl->block_size + next_bl->block_size);

    if (next_bl->_next != NULL) {
        prev_bl->_next = next_bl->_next;
        next_bl->_next->_prev = prev_bl;

        for (int i = dex + 2; i < this->list_length; ++i) {
            this->mem_block_list[i]->_loc.z -= 2;
        }
        this->mem_block_list.erase(
            this->mem_block_list.begin() + dex, this->mem_block_list.begin() + dex + 2);

        free(this_bl);
        free(next_bl);
    }
    else {
        prev_bl->_next = NULL;
        this->mem_block_list.pop_back();
        this->mem_block_list.pop_back();

        free(this_bl);
        free(next_bl);
    }
    return prev_bl;
}


void decx::MemChunk_D::check_to_merge(int dex)
{
    decx::MemBlock* this_bl = *(this->mem_block_list.begin() + dex);
    // The last block, forward_merge2 only
    if (this_bl->_prev != NULL && this_bl->_next == NULL) {
        decx::MemBlock* prev_bl = this_bl->_prev;
        if (prev_bl->_idle) {
            this->forward_merge2(dex);
        }
    }
    // The first block, backward_merge only
    else if (this_bl->_prev == NULL && this_bl->_next != NULL) {
        decx::MemBlock* next_bl = this_bl->_next;
        if (next_bl->_idle) {
            this->backward_merge2(dex);
        }
    }
    else if (this_bl->_prev != NULL && this_bl->_next != NULL) {
        decx::MemBlock* prev_bl = this_bl->_prev;
        decx::MemBlock* next_bl = this_bl->_next;
        if (prev_bl->_idle && (!next_bl->_idle)) {
            this->forward_merge2(dex);
        }
        else if ((!prev_bl->_idle) && next_bl->_idle) {
            this->backward_merge2(dex);
        }
        else if (prev_bl->_idle && next_bl->_idle) {
            this->merge3(dex);
        }
    }
}


decx::MemChunkSet_D::MemChunkSet_D(int pool_dex)
{
    this->mem_chunk_list.emplace_back(pool_dex, 0);
    this->flag_size = 1 << (dex_to_pow_bias + pool_dex);
    this->list_length = this->mem_chunk_list.size();
}