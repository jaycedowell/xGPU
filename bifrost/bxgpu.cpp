/*
 * Copyright (c) 2021, The Bifrost Authors. All rights reserved.
 # Copyright (c) 2021, The University of New Mexico. All rights reserved.

 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#include <bifrost/array.h>
#include <bifrost/common.h>
#include <utils.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "bxgpu.h"

#include "xgpu.h"

//
// Convert from a XGPU data type to a Bifrost data type
//
inline BFdtype bf_dtype_from_xgpu(int dtype) {
    switch(dtype) {
        case XGPU_INT8: return BF_DTYPE_CI8;
        case XGPU_INT32: return BF_DTYPE_CI32;
        default: return BF_DTYPE_CF32;
    }
}

//
// Convert from a XGPU exit code to a Bifrost BFstatus code
//
inline BFstatus bf_status_from_xgpu(int status) {
    switch(status) {
        case XGPU_OUT_OF_MEMORY: // Fall-through
        case XGPU_INSUFFICIENT_TEXTURE_MEMORY: return BF_STATUS_MEM_ALLOC_FAILED;
        case XGPU_CUDA_ERROR: return BF_STATUS_DEVICE_ERROR;
        case XGPU_NOT_INITIALIZED: return BF_STATUS_INVALID_STATE;
        case XGPU_HOST_BUFFER_NOT_SET: return BF_STATUS_INVALID_POINTER;
        case XGPU_CONFIGURATON_ERROR: return BF_STATUS_UNSUPPORTED;
        default: return BF_STATUS_SUCCESS;
    }
}

class bxgpu_impl {
private:
    int _ntime;
    int _nchan;
    int _nstand;
    int _npol;
    int _dev;
    
    int _ready;
    int _is_dp4a;
    XGPUContext _context;
    XGPUInfo _info;
    ComplexInput *_swizzel = NULL;
    cudaStream_t _stream;
public:
    bxgpu_impl() : _ready(0), _is_dp4a(0), _swizzel(NULL), _stream(g_cuda_stream) {}
    ~bxgpu_impl() {
        if(_ready) {
            xgpuFree(&_context);
        }
        if(_swizzel) {
            cudaSetDevice(_dev);
            cudaFree(_swizzel);
        }
    }
    inline int ntime() const { return _ntime; }
    inline int nchan() const { return _nchan; }
    inline int nstand() const { return _nstand; }
    inline int npol() const { return _npol; }
    inline int nbaseline() const { return (int) _info.nbaseline; }
    inline BFdtype in_dtype() const { return bf_dtype_from_xgpu(_info.input_type); }
    inline BFdtype out_dtype() const { return _info.compute_type == XGPU_FLOAT32 ? bf_dtype_from_xgpu(_info.compute_type) : BF_DTYPE_CI32; }
    BFstatus init(int ntime, int nchan, int nstand, int npol, int dev) {
        _ntime = ntime;
        _nchan = nchan;
        _nstand = nstand;
        _npol = npol;
        _dev = dev;
        
        int xgpu_status;
        BFstatus status;
        xgpuInfo(&_info);
        
        BF_ASSERT(_ntime == _info.ntime, BF_STATUS_INVALID_SHAPE);
        BF_ASSERT(_nchan == _info.nfrequency, BF_STATUS_INVALID_SHAPE);
        BF_ASSERT(_nstand == _info.nstation, BF_STATUS_INVALID_SHAPE);
        BF_ASSERT(_npol == _info.npol, BF_STATUS_INVALID_SHAPE);
        
        // We only want XGPU if it is in lower triangular
        BF_ASSERT(TRIANGULAR_ORDER == _info.matrix_order, BF_STATUS_UNSUPPORTED);
        
        // Check for DP4A
        _is_dp4a = (_info.compute_type == XGPU_INT8);
        
        _context.array_h = NULL;
        _context.array_len = _info.vecLength;
        _context.matrix_h = NULL;
        _context.matrix_len = _info.matLength;
        
        xgpu_status = xgpuInit(&_context, _dev | XGPU_DONT_REGISTER);
        status = bf_status_from_xgpu(xgpu_status);
        if(_is_dp4a) {
            cudaSetDevice(_dev);
            
            cudaMalloc((void**)&_swizzel, _info.vecLength*sizeof(ComplexInput));
        }
        xgpuSetStream(&_context, (unsigned long long) _stream);
        
        if( status == BF_STATUS_SUCCESS ) {
            _ready = 1;
        }
        return status;
    }
    BFstatus reset_state() {
        BF_ASSERT(_ready, BF_STATUS_INVALID_STATE); 
        
        int xgpu_status;
        BFstatus status;
        xgpu_status = xgpuClearDeviceIntegrationBuffer(&_context);
        status = bf_status_from_xgpu(xgpu_status);
        return status;
    }
    BFstatus exec(BFarray const* in, BFarray* out, BFbool dump) {
        BF_ASSERT(_ready, BF_STATUS_INVALID_STATE); 
        
        if( _is_dp4a ) {
            xgpuSwizzleInputOnDevice(&_context, _swizzel, (ComplexInput *) in->data);
            _context.array_h = _swizzel;
        } else {
            _context.array_h = (ComplexInput *) in->data;
        }
        xgpuSetHostInputBuffer(&_context);
        _context.matrix_h = (Complex *) out->data;
        xgpuSetHostOutputBuffer(&_context);
        
        int xgpu_status;
        BFstatus status;
        xgpu_status = xgpuCudaXengineOnDevice(&_context,
                                              dump ? SYNCOP_DUMP : SYNCOP_SYNC_COMPUTE);
        status = bf_status_from_xgpu(xgpu_status);
        if( status == BF_STATUS_SUCCESS && dump ) {
            xgpuReorderMatrixOnDevice(&_context, (Complex *) out->data);
            xgpu_status = xgpuClearDeviceIntegrationBuffer(&_context);
            status = bf_status_from_xgpu(xgpu_status);
        }
        return status;
    }
};

BFstatus BXgpuCreate(bxgpu* plan_ptr) {
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new bxgpu_impl(),
                       *plan_ptr = 0);
}

BFstatus BXgpuInit(bxgpu plan,
                   int   ntime,
                   int   nchan,
                   int   nstand,
                   int   npol,
                   int   dev) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    return plan->init(ntime, nchan, nstand, npol, dev);
}

BFstatus BXgpuResetState(bxgpu plan) {
    BF_ASSERT(plan, BF_STATUS_INVALID_POINTER);
    return plan->reset_state();
}

BFstatus BXgpuExecute(bxgpu          plan,
                      BFarray const* in,
                      BFarray*       out,
                      BFbool         dump) {
    BF_ASSERT(plan, BF_STATUS_INVALID_POINTER);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
  	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
  	BF_ASSERT( in->ndim == 3, BF_STATUS_INVALID_SHAPE);
  	BF_ASSERT(out->ndim == 2, BF_STATUS_INVALID_SHAPE);
    
    BF_ASSERT(in->shape[0] == plan->ntime(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[1] == plan->nchan(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[2] == plan->nstand()*plan->npol(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[0] == plan->nchan(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[1] == plan->nbaseline()*plan->npol()*plan->npol(), BF_STATUS_INVALID_SHAPE);
    
    BF_ASSERT(in->dtype == plan->in_dtype(), BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(out->dtype == plan->out_dtype(), BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from(in->space, BF_SPACE_CUDA),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA),
              BF_STATUS_UNSUPPORTED_SPACE);
    
    return plan->exec(in, out, dump);
}

BFstatus BXgpuDestroy(bxgpu plan) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
