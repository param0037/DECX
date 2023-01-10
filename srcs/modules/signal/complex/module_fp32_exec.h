/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _MODULE_FP32_EXEC_H_
#define _MODULE_FP32_EXEC_H_

#include "../../core/thread_management/thread_pool.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../core/thread_management/thread_arrange.h"
#include "../../classes/Matrix.h"
#include "../../classes/classes_util.h"


namespace decx
{
    namespace signal {
        namespace CPUK {
            /**
            * @param src : Pointer of input complex array
            * @param dst : Pointer of output float32 array
            * @param _proc_len : 4x elements
            */
            _THREAD_FUNCTION_ void
            _module_fp32_ST(const double* src, float* dst, const size_t _proc_len);
        }

        /**
        * @param src : Pointer of input complex array
        * @param dst : Pointer of output float32 array
        * @param _total_len : The total length of the 1D array, in de::CPf (already aligned to 4)
        */
        void _module_fp32_caller(const de::CPf* src, float* dst, const size_t _total_len);
    }
}


#endif