/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _SCAN_MODE_
#define _SCAN_MODE_

namespace decx {
    namespace scan {
        enum SCAN_MODE
        {
            SCAN_MODE_INCLUSIVE = 0,
            SCAN_MODE_EXCLUSIVE = 1,

            SCAN_MODE_HORIZONTAL = 2,
            SCAN_MODE_VERTICAL   = 3,
            SCAN_MODE_FULL       = 4
        };
    }
}


#endif