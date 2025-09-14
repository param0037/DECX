// /**
// *   ----------------------------------------------------------------------------------
// *   Author : Wayne Anderson
// *   Date   : 2021.04.16
// *   ----------------------------------------------------------------------------------
// * 
// * This is a part of the open source project named "DECX", a high-performance scientific
// * computational library. This project follows the MIT License. For more information 
// * please visit https://github.com/param0037/DECX.
// * 
// * Copyright (c) 2021 Wayne Anderson
// * 
// * Permission is hereby granted, free of charge, to any person obtaining a copy of this 
// * software and associated documentation files (the "Software"), to deal in the Software 
// * without restriction, including without limitation the rights to use, copy, modify, 
// * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// * permit persons to whom the Software is furnished to do so, subject to the following 
// * conditions:
// * 
// * The above copyright notice and this permission notice shall be included in all copies 
// * or substantial portions of the Software.
// * 
// * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// * DEALINGS IN THE SOFTWARE.
// */

// #include <event.h>


// decx::core::Event::Event()
// {
//     this->_event_flag = 0;
// }


// decx::core::Event::~Event()
// {

// }


// int32_t decx::core::Event::SetEvent(const int32_t event_flag)
// {
//     this->_event_flag |= event_flag;
//     this->_cv.notify_all();
//     return 0;
// }

// namespace decx
// {
// namespace core
// {
//     static int32_t EventWaitSpin_Timeout(Event* p_event, const int32_t event_flag, const uint64_t timeout_msec)
//     {
//         auto start = std::chrono::steady_clock::now();
//         // Check flag first
//         if ((this->GetEventFlag() ^ event_flag) == 0){
//             return (int32_t)EventWaitResult_e::EventWait_Success;
//         }
//         while (1) {
//             if ((this->GetEventFlag() ^ event_flag) == 0){
//                 return (int32_t)EventWaitResult_e::EventWait_Success;
//             }
//             _mm_pause();    // save cycles but keep in scheduler, avoid context switching
//             if (std::chrono::steady_clock::now() - start > timeout) {
//                 return (int32_t)EventWaitResult_e::EventWait_Timeout;
//             }
//         }
//     }


//     static int32_t EventWaitSpin_Forever(Event* p_event, const int32_t event_flag)
//     {
//         auto start = std::chrono::steady_clock::now();
//         // Check flag first
//         if ((this->GetEventFlag() ^ event_flag) == 0){
//             return (int32_t)EventWaitResult_e::EventWait_Success;
//         }
//         while (1) {
//             if ((this->GetEventFlag() ^ event_flag) == 0){
//                 return (int32_t)EventWaitResult_e::EventWait_Success;
//             }
//             _mm_pause();    // save cycles but keep in scheduler, avoid context switching
//         }
//     }


//     static int32_t EventWaitSpin(Event* p_event, const int32_t event_flag, const EventWaitOption_e wait_option, const uint64_t timeout_msec)
//     {
//         switch (wait_option)
//         {
//         case EventWaitOption_e::EventWaitOption_WaitForever:
//             return EventWaitSpin_Forever(p_event, event_flag);
//             break;

//         case EventWaitOption_e::EventWaitOption_WaitTimeout:
//             return EventWaitSpin_Forever(p_event, event_flag, timeout_msec);
//             break;
        
//         default:
//             break;
//         }
//     }


//     static int32_t EventWaitLowpower_Timeout(Event* p_event, const int32_t event_flag, const uint64_t timeout_msec)
//     {
//         auto start = std::chrono::steady_clock::now();
//         // Check flag first
//         if ((this->GetEventFlag() ^ event_flag) == 0){
//             return (int32_t)EventWaitResult_e::EventWait_Success;
//         }
//         while (1) {
            
//         }
//     }
// }
// }

// int32_t 
// decx::core::Event::WaitEvent(const int32_t event_flag, 
//                              const decx::core::EventWaitOption_e wait_option, 
//                              const decx::core::EventWaitBehaviour_e wait_behaviour)
// {

// }