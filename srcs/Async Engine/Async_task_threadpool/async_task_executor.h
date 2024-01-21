/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _ASYNC_TASK_EXECUTOR_H_
#define _ASYNC_TASK_EXECUTOR_H_


#include "../../modules/core/basic.h"
#include "../../modules/core/utils/Dynamic_Array.h"


typedef std::packaged_task<void()> Task;


namespace decx
{
    namespace async {
        class async_task_executor
        {
        public:
            bool _all_done;
            bool _shutdown;
            std::mutex _mtx;
            std::condition_variable _cv;
            uint _task_num;

            std::condition_variable* _cv_main_thread;
            std::mutex* _mtx_main_thread;

            decx::utils::Dynamic_Array<Task> _task_queue;

            std::thread* _exec;

            async_task_executor(std::condition_variable* _cv_main_thread,
                std::mutex* _mtx_main_thread);

            // main_loop callback function running on each thread
            void _thread_main_loop();

            /**
            * Since a std::thread object starts to run a function when it is created.
            * This member function will placement new a std::thread object on the memory and
            * run the main_loop function
            */
            void start();


            void bind_thread_addr(std::thread* _addr);


            void stream_shutdown();


            bool _idle;
            uint _ID;
        };

        typedef decx::async::async_task_executor _ATE;
    }
}



#endif