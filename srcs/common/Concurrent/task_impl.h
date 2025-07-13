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

#ifndef _TASK_IMPL_H_
#define _TASK_IMPL_H_

#include <basic.h>
#include <mutex>
#include <condition_variable>


#ifdef _MSC_VER
#define _THREAD_FUNCTION_  // represents a function that only runs on threads
#define _THREAD_CALL_      // represents a function that is only called by a thread function
#define _THREAD_GENERAL_   // represents a function that can be called within threads and called as a thread function
#endif
#if defined(__GNUC__) || defined(__clang__)
#define _THREAD_FUNCTION_   __attribute__((hot)) // represents a function that only runs on threads
#define _THREAD_CALL_       __attribute__((hot)) // represents a function that is only called by a thread function
#define _THREAD_GENERAL_    __attribute__((hot)) // represents a function that can be called within threads and called as a thread function
#endif


#define TASK_PACK_MAX_SIZE 1024

#if __cplusplus < 201703L
namespace decx
{
namespace utils
{
	template <uint64_t ...idx>
	struct IdxSeq {};


	template <uint64_t N, uint64_t ... idx>
	struct IdxSeqGen : IdxSeqGen<N-1, N-1, idx...> {};


	template <uint64_t... idx>
	struct IdxSeqGen<0, idx...> : IdxSeq<idx...>
	{
		using type = IdxSeq<idx ...>;
	};


	template <typename Tcallable, typename Ttuple, uint64_t... idx>
	auto Apply_Impl(Tcallable&& entry_func, Ttuple&& tuple, IdxSeq<idx...>)
		-> decltype(std::forward<Tcallable>(entry_func)(std::get<idx>(std::forward<Ttuple>(tuple))...))
	{
		return std::forward<Tcallable>(entry_func)(std::get<idx>(std::forward<Ttuple>(tuple))...);
	}


	template <typename Tcallable, typename Ttuple>
	auto Apply(Tcallable&& entry_func, Ttuple&& tuple)
		-> decltype(decx::utils::Apply_Impl(std::forward<Tcallable>(entry_func),
											std::forward<Ttuple>(tuple),
											typename decx::utils::IdxSeqGen<std::tuple_size<typename std::decay<Ttuple>::type>::value>{}
					))
	{
		using TupleType = typename std::decay<Ttuple>::type;
		constexpr uint64_t size = std::tuple_size<TupleType>::value;
		using Indices = typename decx::utils::IdxSeqGen<size>::type;
		return decx::utils::Apply_Impl(std::forward<Tcallable>(entry_func), std::forward<Ttuple>(tuple), Indices{});
	}
}
}
#endif

namespace decx
{
namespace core
{
	class TaskBase;

	template <typename FuncType, typename ... ArgTypes>
	class Task;


    enum class TaskState_e : uint8_t
    {
        TaskState_Idle = 2,
        TaskState_Running = 1,
        TaskState_Pending = 0,
    };

	typedef TaskBase* TaskImplHandle_t;
}
}


class _DECX_API_ decx::core::TaskBase
{
public:
	std::condition_variable _cv;
	std::mutex _mtx;
	TaskState_e _sem;

public:
	virtual void Execute() {}
    void Synchronize();
	virtual ~TaskBase() {}
};


template <typename FuncType, typename ... ArgTypes>
class decx::core::Task : public decx::core::TaskBase
{
private:
	FuncType _task_entry;
	std::tuple<ArgTypes...> _args;

public:
	Task(FuncType task_entry, ArgTypes ... args) : 
		_task_entry(std::forward<FuncType>(task_entry)),
		_args(std::forward<ArgTypes>(args)...)
	{
		this->_sem = decx::core::TaskState_e::TaskState_Pending;
	}


	_THREAD_FUNCTION_ virtual void Execute() override
	{
		std::unique_lock<std::mutex> lock(this->_mtx);
		this->_sem = TaskState_e::TaskState_Running;
#if __cplusplus < 201703L
		decx::utils::Apply(this->_task_entry, this->_args);
#else
		std::apply(this->_task_entry, this->_args);
#endif
		this->_sem = TaskState_e::TaskState_Idle;
		this->_cv.notify_one();
	}
};


#endif