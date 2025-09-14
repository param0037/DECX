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

#ifndef _LOCKFREE_RINGBUFFER_H_
#define _LOCKFREE_RINGBUFFER_H_

#include <basic.h>

namespace decx
{
namespace utils
{
    template <typename _data_type>
    class Lockfree_RingBuffer;


    struct LFQ_Iterator_t
    {
        uint64_t _vir_idx;
    };


    enum class LFQ_Status_e : uint8_t
    {
        LFQ_Success             = 0,        // No error for the operant
        LFQ_Transient           = 1,        // Currently in transient state (Invalid head and end indices)
        LFQ_Eequeue_Occupied    = 2,        // The position to be enqueued is occupied
        LFQ_Dequeue_released    = 3,        // The position to be dequeued is empty
        LFQ_Enqueue_Overflow    = 4,        // No enough space to enqueue the element
        LFQ_Dequeue_Empty       = 5,
    };
}
}


template <typename _data_type>
class decx::utils::Lockfree_RingBuffer
{
private:
    std::atomic<uint64_t> _head_idx;
    std::atomic<uint64_t> _end_idx;

    void* _data;
    std::atomic<uint8_t>* _flag;

    uint64_t _phy_size;
    std::atomic<uint8_t> _operating;

protected:
    void OperatingStatusAcquire()
    {
        uint8_t op_stat_exp = 0;
        do {
            op_stat_exp = 0;
        }
        while(!this->_operating.compare_exchange_strong(op_stat_exp, 
                                                        1, 
                                                        std::memory_order_release, 
                                                        std::memory_order_relaxed));
    }


    int32_t OperatingStatusAcquire_Weak()
    {
        uint8_t op_stat_exp = 0;
        bool res = this->_operating.compare_exchange_strong(op_stat_exp, 
                                                        1, 
                                                        std::memory_order_release, 
                                                        std::memory_order_relaxed);

        return (res) ? 0 : 1;
    }


    void OperatingStatusRelease()
    {
        this->_operating.store(0, std::memory_order_release);
    }


public:
    Lockfree_RingBuffer() 
    {
        this->_head_idx.store(0, std::memory_order_release);
        this->_end_idx.store(0, std::memory_order_release);
        this->_operating.store(0, std::memory_order_release);
        this->_data = nullptr;
        this->_flag = nullptr;
    }


    int32_t Allocate(const uint64_t size)
    {
        this->_phy_size = size;
        this->_data = malloc(this->_phy_size * sizeof(_data_type));
        if (this->_data == nullptr) {
            return -1;
        }
        this->_flag = (std::atomic<uint8_t>*)malloc(this->_phy_size * sizeof(std::atomic<uint8_t>));
        for (uint32_t i = 0; i < this->_phy_size; ++i) {
            this->_flag[i].store(0, std::memory_order_release);
        }
        if (nullptr == this->_flag) {
            return -1;
        }
        // printf("is lock free: %d\n", this->_head_idx.is_lock_free());
        return 0;
    }


    template <typename ...Args>
    LFQ_Status_e Enqueue(Args&& ...args)
    {
        OperatingStatusAcquire();

        uint64_t current_end_idx = 0;
        uint64_t current_head_idx = 0;
        uint64_t phy_widx = 0;
        do {
            current_end_idx = this->_end_idx.load(std::memory_order_acquire);
            current_head_idx = this->_head_idx.load(std::memory_order_acquire);
            if (current_end_idx < current_head_idx || current_end_idx >= current_head_idx + this->_phy_size){
                OperatingStatusRelease();
                return LFQ_Status_e::LFQ_Transient;      // Should wait until that position's data is dequeued ?
            }
            if (this->_flag[phy_widx].load(std::memory_order_acquire)){
                OperatingStatusRelease();
                return LFQ_Status_e::LFQ_Eequeue_Occupied;
            }
            phy_widx = current_end_idx % this->_phy_size;
        }
        while (!this->_end_idx.compare_exchange_strong(current_end_idx, 
                                                       current_end_idx + 1, 
                                                       std::memory_order_release, 
                                                       std::memory_order_relaxed));
        

        new ((_data_type*)this->_data + phy_widx) _data_type {std::forward<Args>(args)...};
        this->_flag[phy_widx].store(1, std::memory_order_seq_cst);

        OperatingStatusRelease();

        return LFQ_Status_e::LFQ_Success;
    }


    uint64_t Size() const
    {
        // return this->_current_size.load(std::memory_order_acquire);
        return 0;
    }


    LFQ_Status_e Dequeue(_data_type* p_data)
    {
        OperatingStatusAcquire();

        uint64_t current_head_idx = 0;
        uint64_t current_end_idx = 0;
        uint64_t phy_ridx = 0;

        do {
            current_head_idx = this->_head_idx.load(std::memory_order_acquire);
            current_end_idx = this->_end_idx.load(std::memory_order_acquire);
            if (current_end_idx < current_head_idx || current_end_idx >= current_head_idx + this->_phy_size){
                OperatingStatusRelease();
                return LFQ_Status_e::LFQ_Transient;      // Should wait until that position's data is dequeued ?
            }
            phy_ridx = current_head_idx % this->_phy_size;
            if (!this->_flag[phy_ridx].load(std::memory_order_acquire)){
                OperatingStatusRelease();
                return LFQ_Status_e::LFQ_Dequeue_released;
            }
        }
        while (!this->_head_idx.compare_exchange_strong(current_head_idx, 
                                                        current_head_idx + 1, 
                                                        std::memory_order_release, 
                                                        std::memory_order_relaxed));
        *p_data = (*((_data_type*)this->_data + phy_ridx));
        this->_flag[phy_ridx].store(0, std::memory_order_release);

        std::atomic_thread_fence(std::memory_order_seq_cst);

        OperatingStatusRelease();

        return LFQ_Status_e::LFQ_Success;
    }


    LFQ_Status_e PopBack(_data_type* p_data)
    {
        OperatingStatusAcquire();
        
        uint64_t current_head_idx = 0;
        uint64_t current_end_idx = 0;
        uint64_t phy_widx = 0;

        do {
            current_head_idx = this->_head_idx.load(std::memory_order_relaxed);
            current_end_idx = this->_end_idx.load(std::memory_order_relaxed);
            if (current_end_idx == current_head_idx){
                OperatingStatusRelease();
                return LFQ_Status_e::LFQ_Dequeue_Empty;
            }
            if (current_end_idx < current_head_idx || current_end_idx >= current_head_idx + this->_phy_size){
                OperatingStatusRelease();
                return LFQ_Status_e::LFQ_Transient;      // Should wait until that position's data is dequeued ?
            }
            phy_widx = (current_end_idx - 1) % this->_phy_size;
            if (!this->_flag[phy_widx].load(std::memory_order_relaxed)){
                OperatingStatusRelease();
                return LFQ_Status_e::LFQ_Dequeue_released;
            }
        }
        while (!this->_end_idx.compare_exchange_strong(current_end_idx, 
                                                       current_end_idx - 1, 
                                                       std::memory_order_release, 
                                                       std::memory_order_relaxed));

        std::atomic_thread_fence(std::memory_order_seq_cst);

        *p_data = (*((_data_type*)this->_data + phy_widx));
        this->_flag[phy_widx].store(0, std::memory_order_release);

        OperatingStatusRelease();

        return LFQ_Status_e::LFQ_Success;
    }


    LFQ_Status_e TryDequeue(_data_type* p_data)
    {
        if (OperatingStatusAcquire_Weak()){
            return LFQ_Status_e::LFQ_Transient;
        }

        uint64_t current_head_idx = this->_head_idx.load(std::memory_order_acquire);
        uint64_t current_end_idx = this->_end_idx.load(std::memory_order_acquire);
        uint64_t phy_ridx = 0;

        if (current_end_idx < current_head_idx || current_end_idx >= current_head_idx + this->_phy_size){
            OperatingStatusRelease();
            return LFQ_Status_e::LFQ_Transient;      // Should wait until that position's data is dequeued ?
        }
        phy_ridx = current_head_idx % this->_phy_size;
        if (!this->_flag[phy_ridx].load(std::memory_order_acquire)){
            OperatingStatusRelease();
            return LFQ_Status_e::LFQ_Dequeue_released;
        }
        if (!this->_head_idx.compare_exchange_strong(current_head_idx, 
                                                     current_head_idx + 1, 
                                                     std::memory_order_release, 
                                                     std::memory_order_relaxed)){
                                                        OperatingStatusRelease();
                                                        return LFQ_Status_e::LFQ_Transient;
                                                    }
        *p_data = (*((_data_type*)this->_data + phy_ridx));
        this->_flag[phy_ridx].store(0, std::memory_order_release);

        OperatingStatusRelease();

        return LFQ_Status_e::LFQ_Success;
    }
};


#endif