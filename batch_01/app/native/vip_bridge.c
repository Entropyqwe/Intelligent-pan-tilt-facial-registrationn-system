/* Small application-owned ABI compiled against the TARGET's official VIPLite
 * headers. No guessed vendor structures are exported to Python. */
#include <vip_lite.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define FC_MAX_IO 16
typedef struct {
    uint32_t dimensions, sizes[6], format, quantization;
    int32_t fixed_point, zero_point;
    float scale;
    uint32_t bytes, elements;
} fc_tensor_info;

typedef struct {
    vip_network network;
    vip_buffer inputs[FC_MAX_IO], outputs[FC_MAX_IO];
    void *input_map[FC_MAX_IO], *output_map[FC_MAX_IO];
    fc_tensor_info in_info[FC_MAX_IO], out_info[FC_MAX_IO];
    uint32_t input_count, output_count;
    int initialized, prepared;
} fc_context;

void fc_close(fc_context *c) {
    uint32_t i;
    if (!c) return;
    if (c->prepared) vip_finish_network(c->network);
    for (i=0; i<FC_MAX_IO; ++i) {
        if (c->input_map[i]) vip_unmap_buffer(c->inputs[i]);
        if (c->output_map[i]) vip_unmap_buffer(c->outputs[i]);
        if (c->inputs[i]) vip_destroy_buffer(c->inputs[i]);
        if (c->outputs[i]) vip_destroy_buffer(c->outputs[i]);
    }
    if (c->network) vip_destroy_network(c->network);
    if (c->initialized) vip_destroy();
    free(c);
}

static int create_io(fc_context *c, int output, uint32_t index) {
    vip_buffer_create_params_t p;
    fc_tensor_info *info = output ? &c->out_info[index] : &c->in_info[index];
    vip_buffer *buffer = output ? &c->outputs[index] : &c->inputs[index];
    void **mapped = output ? &c->output_map[index] : &c->input_map[index];
    vip_status_e status;
    uint32_t i;
    int64_t elements=1;
    memset(&p, 0, sizeof(p));
#define QUERY(property, address) do { \
    status = output ? vip_query_output(c->network,index,property,address) \
                    : vip_query_input(c->network,index,property,address); \
    if (status != VIP_SUCCESS) return (int)status; \
} while (0)
    QUERY(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &p.num_of_dims);
    if (!p.num_of_dims || p.num_of_dims > 6) return -1001;
    QUERY(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p.sizes);
    QUERY(VIP_BUFFER_PROP_DATA_FORMAT, &p.data_format);
    QUERY(VIP_BUFFER_PROP_QUANT_FORMAT, &p.quant_format);
    if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        QUERY(VIP_BUFFER_PROP_TF_SCALE, &p.quant_data.affine.scale);
        QUERY(VIP_BUFFER_PROP_TF_ZERO_POINT, &p.quant_data.affine.zeroPoint);
        info->scale=p.quant_data.affine.scale;
        info->zero_point=p.quant_data.affine.zeroPoint;
    } else if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        uint8_t fixed=0;
        QUERY(VIP_BUFFER_PROP_FIXED_POINT_POS, &fixed);
        p.quant_data.dfp.fixed_point_pos=(int8_t)fixed;
        info->fixed_point=(int8_t)fixed;
    } else if (p.quant_format != VIP_BUFFER_QUANTIZE_NONE) return -1002;
    info->dimensions=p.num_of_dims;
    info->format=p.data_format;
    info->quantization=p.quant_format;
    for(i=0;i<p.num_of_dims;++i) {
        info->sizes[i]=p.sizes[i];
        if (!p.sizes[i] || elements > 100000000 / p.sizes[i]) return -1003;
        elements*=p.sizes[i];
    }
    info->elements=(uint32_t)elements;
    status=vip_create_buffer(&p,sizeof(p),buffer);
    if(status!=VIP_SUCCESS) return (int)status;
    info->bytes=vip_get_buffer_size(*buffer);
    *mapped=vip_map_buffer(*buffer);
    return *mapped ? 0 : -1004;
#undef QUERY
}

fc_context *fc_open(const char *path, char *error, uint32_t error_size) {
    fc_context *c=(fc_context*)calloc(1,sizeof(fc_context));
    int status=-1000;
    uint32_t i;
    const char *stage="allocation";
    if(!c) goto fail;
#define CALL(operation) do { stage=#operation; status=(int)(operation); if(status) goto fail; } while(0)
    CALL(vip_init()); c->initialized=1;
    CALL(vip_create_network(path,0,VIP_CREATE_NETWORK_FROM_FILE,&c->network));
    CALL(vip_query_network(c->network,VIP_NETWORK_PROP_INPUT_COUNT,&c->input_count));
    CALL(vip_query_network(c->network,VIP_NETWORK_PROP_OUTPUT_COUNT,&c->output_count));
    if(c->input_count!=1 || !c->output_count || c->output_count>FC_MAX_IO) {
        stage="input/output count validation"; status=-1005; goto fail;
    }
    for(i=0;i<c->input_count;++i) CALL(create_io(c,0,i));
    for(i=0;i<c->output_count;++i) CALL(create_io(c,1,i));
    CALL(vip_prepare_network(c->network)); c->prepared=1;
    for(i=0;i<c->input_count;++i) CALL(vip_set_input(c->network,i,c->inputs[i]));
    for(i=0;i<c->output_count;++i) CALL(vip_set_output(c->network,i,c->outputs[i]));
    return c;
fail:
    if(error && error_size) snprintf(error,error_size,"%s failed (VIPLite status %d)",stage,status);
    fc_close(c);
    return NULL;
#undef CALL
}

int fc_info(fc_context *c, int output, uint32_t index, fc_tensor_info *info) {
    if(!c || !info || index >= (output ? c->output_count : c->input_count)) return -1;
    *info=output ? c->out_info[index] : c->in_info[index];
    return 0;
}

uint32_t fc_output_count(fc_context *c) { return c ? c->output_count : 0; }

int fc_run(fc_context *c, const void *input, uint32_t bytes) {
    int status;
    if(!c || !input || bytes!=c->in_info[0].bytes) return -1006;
    memcpy(c->input_map[0],input,bytes);
    status=(int)vip_flush_buffer(c->inputs[0],VIP_BUFFER_OPER_TYPE_FLUSH);
    if(status) return status;
    /* Keep the prepared network alive across frames; finish only on close. */
    return (int)vip_run_network(c->network);
}

static float half_to_float(uint16_t bits) {
    unsigned sign=bits>>15, exponent=(bits>>10)&31, fraction=bits&1023;
    float f;
    if(exponent==0) f=ldexpf((float)fraction,-24);
    else if(exponent==31) f=fraction ? NAN : INFINITY;
    else f=ldexpf((float)(1024+fraction),(int)exponent-25);
    return sign ? -f : f;
}

int fc_read_output(fc_context *c, uint32_t index, float *destination, uint32_t elements) {
    uint32_t i;
    fc_tensor_info *info;
    const void *source;
    float scale=1.0f,zero=0.0f;
    int status;
    if(!c || index>=c->output_count || !destination) return -1007;
    info=&c->out_info[index];
    if(elements!=info->elements) return -1008;
    status=(int)vip_flush_buffer(c->outputs[index],VIP_BUFFER_OPER_TYPE_INVALIDATE);
    if(status) return status;
    source=c->output_map[index];
    if(info->format==VIP_BUFFER_FORMAT_FP32 && info->quantization==VIP_BUFFER_QUANTIZE_NONE) {
        memcpy(destination,source,(size_t)elements*sizeof(float));
        return 0;
    }
    if(info->quantization==VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        scale=info->scale; zero=(float)info->zero_point;
    } else if(info->quantization==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        scale=ldexpf(1.0f,-info->fixed_point);
    }
    for(i=0;i<elements;++i) {
        float value;
        switch(info->format) {
            case VIP_BUFFER_FORMAT_FP32: value=((const float*)source)[i]; break;
            case VIP_BUFFER_FORMAT_FP16: value=half_to_float(((const uint16_t*)source)[i]); break;
            case VIP_BUFFER_FORMAT_UINT8: value=((const uint8_t*)source)[i]; break;
            case VIP_BUFFER_FORMAT_INT8: value=((const int8_t*)source)[i]; break;
            case VIP_BUFFER_FORMAT_INT16: value=((const int16_t*)source)[i]; break;
            case VIP_BUFFER_FORMAT_UINT16: value=((const uint16_t*)source)[i]; break;
            case VIP_BUFFER_FORMAT_INT32: value=(float)((const int32_t*)source)[i]; break;
            default: return -1009;
        }
        destination[i]=(value-zero)*scale;
    }
    return 0;
}

/* Compact the supported YOLOv5 float heads in-place at mapped NPU output.
 * Rows: stride,anchor,grid_y,grid_x,tx,ty,tw,th,confidence. This avoids
 * copying all 80 class channels to Python for every grid point. */
int fc_person_candidates(fc_context *c, float threshold, float *rows, uint32_t capacity) {
    uint32_t output,a,y,x,k,count=0;
    float gate;
    if(!c || !rows || !(threshold>0.0f && threshold<1.0f) || c->output_count!=3) return -1010;
    gate=logf(threshold/(1.0f-threshold));
    for(output=0;output<c->output_count;++output) {
        fc_tensor_info *info=&c->out_info[output];
        const float *data=(const float*)c->output_map[output];
        uint32_t grid, stride;
        int channel_first;
        int status;
        if(info->format!=VIP_BUFFER_FORMAT_FP32 || info->quantization!=VIP_BUFFER_QUANTIZE_NONE ||
           info->dimensions!=5 || info->sizes[3]!=3 || info->sizes[4]!=1) return -1011;
        channel_first=(info->sizes[2]==85 && info->sizes[0]==info->sizes[1]);
        grid=channel_first ? info->sizes[0] : info->sizes[1];
        if((!channel_first && (info->sizes[0]!=85 || info->sizes[2]!=grid)) || !grid || 640%grid) return -1011;
        stride=640/grid;
        if(stride!=8 && stride!=16 && stride!=32) return -1012;
        status=(int)vip_flush_buffer(c->outputs[output],VIP_BUFFER_OPER_TYPE_INVALIDATE);
        if(status) return status;
        for(a=0;a<3;++a) for(y=0;y<grid;++y) for(x=0;x<grid;++x) {
            uint32_t cell=y*grid+x, plane=grid*grid;
            const float *v=channel_first ? data+a*85*plane+cell : data+((a*grid+y)*grid+x)*85;
            uint32_t step=channel_first ? plane : 1;
            float score;
            int person=1;
            if(v[4*step]<gate || v[5*step]<gate || !isfinite(v[4*step]) || !isfinite(v[5*step])) continue;
            score=(1.0f/(1.0f+expf(-v[4*step])))*(1.0f/(1.0f+expf(-v[5*step])));
            if(score<threshold) continue;
            for(k=6;k<85;++k) if(!isfinite(v[k*step]) || v[k*step]>v[5*step]) {person=0;break;}
            if(!person) continue;
            for(k=0;k<4;++k) if(!isfinite(v[k*step])) return -1013;
            /* Never silently truncate candidates: fail closed instead. */
            if(count>=capacity) return -1014;
            rows[9*count]=(float)stride; rows[9*count+1]=(float)a;
            rows[9*count+2]=(float)y; rows[9*count+3]=(float)x;
            for(k=0;k<4;++k) rows[9*count+4+k]=v[k*step];
            rows[9*count+8]=score;
            ++count;
        }
    }
    return (int)count;
}
