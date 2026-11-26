/* SPDX-License-Identifier: MIT */
/* Host admission rejects missing Header before callbacks. Corrupted valid
 * Headers exercise scientific abandonment; explicit reset recovers host rejection. */
static void test_streaming_metadata_declarations(const char *plugin)
{
    char config[2200];struct spa_fgn_graph *graph=NULL;
    int n=snprintf(config,sizeof(config),
        "{ nodes=[ { type=ndarray name=r plugin=\"%s\" label = incremental-dense-reconstructor-f32"
        " config = { measurement_count=2 measurement_block_capacity=1 blocks_per_sample=2 reconstructed_count=1 initial_reconstructor=[-0.5 1.5]"
        " measurements_schema=test.measurements/1 column_indices_schema=test.columns/1 measurement_count_schema=test.count/1 reconstructed_schema=test.output/1 reconstructor_schema=test.matrix/1 rate=[10 1] } } ]"
        " inputs=[\"r:measurements\" \"r:columns\" \"r:count\"] outputs=[\"r:reconstructed\"] }",plugin);
    assert(n>0 && (size_t)n<sizeof(config));
    int prepared=spa_fgn_graph_new(config,&graph);
    if(prepared<0)fprintf(stderr,"streaming metadata graph: %s (%d)\n%s\n",strerror(-prepared),prepared,config);
    assert(prepared==0);
    float value[1]={1},result[1]={71};uint32_t column[1]={0},count[1]={1};
    struct raw_test_buffer input,columns,counts,output;
    struct spa_buffer *inputs[]={&input.buffer,&columns.buffer,&counts.buffer},*outputs[]={&output.buffer};
    init_raw_buffer(&input,value,sizeof(value));init_raw_buffer(&columns,column,sizeof(column));
    init_raw_buffer(&counts,count,sizeof(count));init_raw_buffer(&output,result,sizeof(result));
    input.header.seq=31;assert(spa_fgn_graph_activate(graph)==0);
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);assert(output.chunk.size==0);
    input.header.offset=1;input.header.flags=SPA_META_HEADER_FLAG_MARKER|SPA_META_HEADER_FLAG_CORRUPTED;column[0]=1;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);assert(output.chunk.size==0 && result[0]==71);
    input.header.offset=0;input.header.flags=0;column[0]=0;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);assert(output.chunk.size==0);
    input.header.offset=1;input.header.flags=SPA_META_HEADER_FLAG_MARKER;column[0]=1;input.buffer.n_metas=0;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==-EINVAL);
    assert(output.chunk.size==0);
    assert(spa_fgn_graph_reset(graph)==0);
    input.buffer.n_metas=1;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    assert(output.chunk.size==0 && result[0]==71);
    input.header.seq=32;input.header.offset=0;input.header.flags=0;column[0]=0;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    input.header.offset=1;input.header.flags=SPA_META_HEADER_FLAG_MARKER;column[0]=1;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    assert(output.chunk.size==sizeof(result) && result[0]==1.0f && output.header.seq==32);
    assert(output.header.flags & SPA_META_HEADER_FLAG_DISCONT);
    value[0]=FLT_MAX;column[0]=0;input.header.seq=33;input.header.offset=0;input.header.flags=0;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    column[0]=1;input.header.offset=1;input.header.flags=SPA_META_HEADER_FLAG_MARKER;
    /* The fused -0.5*MAX + 1.5*MAX update is finite in current JFG. */
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    assert(result[0]==FLT_MAX && output.header.seq==33);
    /* A genuinely nonfinite fused result still withholds data and metadata. */
    value[0]=0;column[0]=0;input.header.seq=34;input.header.offset=0;input.header.flags=0;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    value[0]=FLT_MAX;column[0]=1;input.header.offset=1;input.header.flags=SPA_META_HEADER_FLAG_MARKER;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)<0);
    assert(result[0]==FLT_MAX && output.header.seq==0);
    value[0]=1;column[0]=0;input.header.seq=35;input.header.offset=0;input.header.flags=0;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    column[0]=1;input.header.offset=1;input.header.flags=SPA_META_HEADER_FLAG_MARKER;
    assert(spa_fgn_graph_process(graph,inputs,3,outputs,1)==0);
    assert(result[0]==1.0f && output.header.seq==35);
    assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);

    n=snprintf(config,sizeof(config),
        "{ nodes=[ { type=ndarray name=p plugin=\"%s\" label = pwfs-row-reconstructor-f32"
        " config = { image_size = [ 4 2 ]  row_block_rows=1 pupil_size = [ 1 1"
        " ]  pupil_origins=[[0 0] [1 0] [2 0] [3 0]] pupil_mask=[true] reconstructed_count=1"
        " initial_reconstructor=[1 0 0 0] reconstructed_schema=test.output/1 rate=[10 1] } } ]"
        " inputs=[\"p:row-block\"] outputs=[\"p:reconstruction-pixels\" \"p:mean-pupil-intensity\" \"p:reconstructed\"] }",plugin);
    assert(n>0 && (size_t)n<sizeof(config));assert(spa_fgn_graph_new(config,&graph)==0);
    float block[2]={2,2},pixels[4]={71,71,71,71},mean[1]={71},reconstructed[1]={71};
    struct raw_test_buffer block_buffer,pixel_buffer,mean_buffer,reconstructed_buffer;
    struct spa_buffer *p_inputs[]={&block_buffer.buffer};
    struct spa_buffer *p_outputs[]={&pixel_buffer.buffer,&mean_buffer.buffer,&reconstructed_buffer.buffer};
    init_raw_buffer(&block_buffer,block,sizeof(block));init_raw_buffer(&pixel_buffer,pixels,sizeof(pixels));
    init_raw_buffer(&mean_buffer,mean,sizeof(mean));init_raw_buffer(&reconstructed_buffer,reconstructed,sizeof(reconstructed));
    block_buffer.header.seq=41;assert(spa_fgn_graph_activate(graph)==0);
    assert(spa_fgn_graph_process(graph,p_inputs,1,p_outputs,3)==0);
    block_buffer.header.offset=1;block_buffer.header.flags=SPA_META_HEADER_FLAG_CORRUPTED;
    assert(spa_fgn_graph_process(graph,p_inputs,1,p_outputs,3)==0);
    assert(pixel_buffer.chunk.size==0 && mean_buffer.chunk.size==0 && reconstructed_buffer.chunk.size==0);
    block_buffer.header.offset=0;block_buffer.header.flags=0;
    assert(spa_fgn_graph_process(graph,p_inputs,1,p_outputs,3)==0);
    block_buffer.header.offset=1;block_buffer.buffer.n_metas=0;
    assert(spa_fgn_graph_process(graph,p_inputs,1,p_outputs,3)==-EINVAL);
    assert(pixel_buffer.chunk.size==0 && mean_buffer.chunk.size==0 && reconstructed_buffer.chunk.size==0);
    assert(spa_fgn_graph_reset(graph)==0);
    block_buffer.buffer.n_metas=1;
    for (uint32_t row=1;row<4;row++) {
        block_buffer.header.offset=row;block_buffer.header.flags=row==3 ? SPA_META_HEADER_FLAG_MARKER : 0;
        assert(spa_fgn_graph_process(graph,p_inputs,1,p_outputs,3)==0);
        assert(pixel_buffer.chunk.size==0 && mean_buffer.chunk.size==0 && reconstructed_buffer.chunk.size==0);
    }
    assert(mean[0]==71 && reconstructed[0]==71);
    block_buffer.header.seq=42;
    for (uint32_t row=0;row<4;row++) {
        block_buffer.header.offset=row;block_buffer.header.flags=row==3 ? SPA_META_HEADER_FLAG_MARKER : 0;
        assert(spa_fgn_graph_process(graph,p_inputs,1,p_outputs,3)==0);
    }
    assert(mean[0]==2 && reconstructed[0]==0 && mean_buffer.header.seq==42);
    assert(mean_buffer.header.flags & SPA_META_HEADER_FLAG_DISCONT);
    assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);
}
