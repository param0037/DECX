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

#ifndef _THREAD_ARGUMENT_H_
#define _THREAD_ARGUMENT_H_

#include <basic.h>


namespace decx
{
    template <typename _data_type, typename FuncType> class ThreadArg_var;

    template <typename _data_type> class ThreadArg_still;
}


template <typename _data_type, typename FuncType>
class decx::ThreadArg_var
{
private:
    using T_updator = _data_type(const int32_t);

    _data_type _arg;
    FuncType _updator;

public:
    ThreadArg_var(_data_type arg, FuncType updator) :
        _arg(arg), _updator(updator) {}


    ThreadArg_var(FuncType updator) :
        _updator(updator) {}


    template <typename ...Args>
    _data_type value(const Args... __args) {
        return this->_updator(__args...);
    }
};


template <typename _data_type>
class decx::ThreadArg_still
{
private:
    _data_type _val;

public:
    ThreadArg_still(_data_type val): _val(val) {}


    template <typename ...Args>
    _data_type value(const Args... __args) {
        return _val;
    }
};


namespace decx
{
    template <typename _data_type, typename LambdaFunc_T>
    inline decx::ThreadArg_var<_data_type, LambdaFunc_T> TArg_var(_data_type val, LambdaFunc_T func){
        return decx::ThreadArg_var<_data_type, LambdaFunc_T>(val, func);
    }


    template <typename _data_type, typename LambdaFunc_T>
    inline decx::ThreadArg_var<_data_type, LambdaFunc_T> TArg_var(LambdaFunc_T func){
        return decx::ThreadArg_var<_data_type, LambdaFunc_T>(func);
    }


    template <typename _data_type>
    inline decx::ThreadArg_still<_data_type> TArg_still(_data_type val){
        return decx::ThreadArg_still<_data_type>(val);
    }
}

#define _TARG_PTR_INC_(ptr, inc) [&](const int32_t i){return (ptr) + i * (inc);}
#define _TARG_PTR_DEC_(ptr, inc) [&](const int32_t i){return (ptr) - i * (inc);}


#endif
