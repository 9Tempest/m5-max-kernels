using namespace metal;
constexpr uint width=$WIDTH, block=$BLOCK, threads=$THREADS;
constexpr uint groups=threads/32, registers=block/32;
constexpr uint blocks=width/block, slots=(blocks+groups-1)/groups;
constexpr bool layer=$LAYER, two_read=$TWO_READ;
constexpr float epsilon=$EPSILON;
uint tid=thread_position_in_threadgroup.x;
uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
uint row=threadgroup_position_in_grid.x;
uint modulation_offset=$MOD_OFFSET;
$GATE_OFFSET
threadgroup float partials[groups];
float values[slots][registers];
float first=0.0f;

// Each SIMD group owns complete blocks, including the final partial group
// of blocks. Invalid slots contribute zero and never read or write global X.
#pragma unroll
for(uint slot=0;slot<slots;++slot) {
    uint b=slot*groups+sg;
    #pragma unroll
    for(uint reg=0;reg<registers;++reg) {
        uint j=b*block+reg*32+lane;
        float v=b<blocks ? $LOAD_X : 0.0f;
        $WRITE_RESIDUAL
        if(!two_read) values[slot][reg]=v;
        first+=layer ? v : v*v;
    }
}
first=simd_sum(first);
if(lane==0) partials[sg]=first;
threadgroup_barrier(mem_flags::mem_threadgroup);
first=simd_sum(lane<groups ? partials[lane] : 0.0f)/float(width);
// Reuse of partials is safe only after every SIMD group has read it.
threadgroup_barrier(mem_flags::mem_threadgroup);

float mean=layer ? first : 0.0f;
float variance=first;
if(layer) {
    variance=0.0f;
    #pragma unroll
    for(uint slot=0;slot<slots;++slot) {
        uint b=slot*groups+sg;
        #pragma unroll
        for(uint reg=0;reg<registers;++reg) {
            uint j=b*block+reg*32+lane;
            float v=two_read ? (b<blocks ? $LOAD_X : 0.0f)
                             : values[slot][reg];
            values[slot][reg]=v;
            float centered=b<blocks ? v-mean : 0.0f;
            variance+=centered*centered;
        }
    }
    variance=simd_sum(variance);
    if(lane==0) partials[sg]=variance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    variance=simd_sum(lane<groups ? partials[lane] : 0.0f)/float(width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
float inv=rsqrt(variance+epsilon), peak=0.0f;

#pragma unroll
for(uint slot=0;slot<slots;++slot) {
    uint b=slot*groups+sg;
    #pragma unroll
    for(uint reg=0;reg<registers;++reg) {
        uint j=b*block+reg*32+lane;
        float v=0.0f;
        if(b<blocks) {
            float x=(two_read&&!layer) ? $LOAD_X : values[slot][reg];
            float normalized=(x-mean)*inv*float(Gamma[j]);
            float modulated=normalized*(1.0f+float(Scale[modulation_offset+j]))
                            +float(Shift[modulation_offset+j]);
            // Match the staged BF16/FP16 boundary after full modulation.
            v=float(O(modulated))*float(Signs[j]);
        }
        #pragma unroll
        for(uint stride=1;stride<32;stride*=2) {
            float peer=simd_shuffle_xor(v,stride);
            v=(lane&stride) ? peer-v : v+peer;
        }
        values[slot][reg]=v;
    }
    // Dimensions above 32 stay within each lane's register array. A lane
    // holds positions lane, lane+32, ... of a complete block, so no shared
    // memory or threadgroup barriers are needed for these butterflies.
    #pragma unroll
    for(uint stride=1;stride<registers;stride*=2) {
        #pragma unroll
        for(uint base=0;base<registers;base+=2*stride) {
            #pragma unroll
            for(uint j=0;j<stride;++j) {
                float a=values[slot][base+j], c=values[slot][base+j+stride];
                values[slot][base+j]=a+c;
                values[slot][base+j+stride]=a-c;
            }
        }
    }
    #pragma unroll
    for(uint reg=0;reg<registers;++reg) {
        values[slot][reg]*=rsqrt(float(block));
        peak=max(peak,abs(values[slot][reg]));
    }
}
peak=simd_max(peak);
if(lane==0) partials[sg]=peak;
threadgroup_barrier(mem_flags::mem_threadgroup);
peak=simd_max(lane<groups ? partials[lane] : 0.0f);
float scale=peak==0.0f ? 1.0f : peak/127.0f;
if(tid==0) S[row]=scale;
#pragma unroll
for(uint slot=0;slot<slots;++slot) {
    uint b=slot*groups+sg;
    if(b<blocks) {
        #pragma unroll
        for(uint reg=0;reg<registers;++reg) {
            uint j=b*block+reg*32+lane;
            Q[row*width+j]=char(clamp(rint(values[slot][reg]/scale),-127.0f,127.0f));
        }
    }
}
