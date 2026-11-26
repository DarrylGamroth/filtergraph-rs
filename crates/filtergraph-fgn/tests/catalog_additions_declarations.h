/* SPDX-License-Identifier: MIT */
/* Current JFG catalog additions, exercised through the actual C graph owner. */
static void test_catalog_pixel_validity(const char *plugin)
{
 struct spa_fgn_graph *graph=NULL; char config[1400];
 float x[4]={0.0f,NAN,INFINITY,-3.0f}; uint8_t valid[4]={0};
 struct raw_test_buffer input,output;struct spa_buffer *inputs[]={&input.buffer},*outputs[]={&output.buffer};
 int res=snprintf(config,sizeof(config),"{ nodes = [ { type = ndarray name = v plugin = \"%s\" label = pixel-validity-f32 config = { extent = 4 input_schema = test.pixel/1 } } ] inputs = [ \"v:input\" ] outputs = [ \"v:validity\" ] }",plugin);
 assert(res>0&&(size_t)res<sizeof(config));assert(spa_fgn_graph_new(config,&graph)==0);
 init_raw_buffer(&input,x,sizeof(x));init_raw_buffer(&output,valid,sizeof(valid));input.header.seq=321;
 assert(spa_fgn_graph_activate(graph)==0);assert(spa_fgn_graph_process(graph,inputs,1,outputs,1)==0);
 assert(valid[0]==1&&valid[1]==0&&valid[2]==0&&valid[3]==1&&output.header.seq==321);
 assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);
}
static void test_catalog_physical_offset(const char *plugin)
{
 struct spa_fgn_graph *graph=NULL;char config[2000];float x[3]={1.0f,-2.0f,0.125f},q[4]={1,2,-0.5,3},c[6]={1,-2,0.5,1,-3,0.25},offset[2]={0.25,-0.5},y[3]={71,71,71},physical[2]={71,71};
 struct raw_test_buffer in,qbuf,cbuf,ob,out,telemetry;struct spa_buffer *inputs[]={&in.buffer,NULL,NULL,NULL},*outputs[]={&out.buffer,&telemetry.buffer};
 int res=snprintf(config,sizeof(config),"{ nodes = [ { type = ndarray name = p plugin = \"%s\" label = physical-offset-reference-f32 config = { residual_count = 3 physical_count = 2 mode_count = 2 reference_sign = -1 residual_schema = test.residual/1 physical_schema = test.physical/1 physical_to_modes_schema = test.physical-to-modes/1 modes_to_reference_schema = test.modes-to-reference/1 } } ] inputs = [ \"p:reconstructed-residual\" \"p:offset\" \"p:physical-to-modes\" \"p:modes-to-reference\" ] outputs = [ \"p:compensated-residual\" \"p:physical-offset\" ] }",plugin);
 assert(res>0&&(size_t)res<sizeof(config));assert(spa_fgn_graph_new(config,&graph)==0);
 init_raw_buffer(&in,x,sizeof(x));init_raw_buffer(&qbuf,q,sizeof(q));init_raw_buffer(&cbuf,c,sizeof(c));init_raw_buffer(&ob,offset,sizeof(offset));init_raw_buffer(&out,y,sizeof(y));init_raw_buffer(&telemetry,physical,sizeof(physical));in.header.seq=322;
 assert(spa_fgn_graph_activate(graph)==0);assert(spa_fgn_graph_update_parameter(graph,2,&qbuf.buffer)==0);
 assert(spa_fgn_graph_process(graph,inputs,4,outputs,2)==SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
 assert(spa_fgn_graph_update_parameter(graph,3,&cbuf.buffer)==0);
 assert(spa_fgn_graph_process(graph,inputs,4,outputs,2)==SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
 assert(spa_fgn_graph_update_parameter(graph,1,&ob.buffer)==0);
 assert(spa_fgn_graph_process(graph,inputs,4,outputs,2)==SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
 assert(y[0]==-1.5f&&y[1]==0.0f&&y[2]==-1.71875f&&physical[0]==0.25f&&physical[1]==-0.5f);assert(out.header.seq==322&&telemetry.header.seq==322);
 offset[0]=NAN;assert(spa_fgn_graph_update_parameter(graph,1,&ob.buffer)<0);assert(spa_fgn_graph_process(graph,inputs,4,outputs,2)==0);assert(physical[0]==0.25f);
 x[1]=NAN;in.header.seq=323;assert(spa_fgn_graph_process(graph,inputs,4,outputs,2)<0);assert(y[0]==-1.5f&&y[1]==0.0f&&y[2]==-1.71875f&&physical[0]==0.25f&&physical[1]==-0.5f);assert(out.header.seq==0&&telemetry.header.seq==0);
 x[1]=-2;assert(spa_fgn_graph_reset(graph)==0);assert(spa_fgn_graph_process(graph,inputs,4,outputs,2)==0);assert(y[0]==-1.5f&&physical[0]==0.25f&&out.header.seq==323);
 assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);
}
static void test_catalog_array_statistics(const char *plugin)
{
 const char *modes[]={"finite","sliding","exponential"};
 for(unsigned mode=0;mode<3;mode++){
  struct spa_fgn_graph *graph=NULL;char config[2000];float x[4]={1,2,3,4},mean[4]={0},variance[4]={0},sd[4]={0};uint64_t count=0;uint8_t complete=0,valid=0;
  struct raw_test_buffer in,a,b,c,d,e,f;struct spa_buffer *inputs[]={&in.buffer},*outputs[]={&a.buffer,&b.buffer,&c.buffer,&d.buffer,&e.buffer,&f.buffer};
  int res=snprintf(config,sizeof(config),"{ nodes = [ { type = ndarray name = s plugin = \"%s\" label = array-statistics-f32 config = { size = [ 2 2 ] mode = %s window_samples = 2 input_schema = test.sample/1 mean_schema = test.mean/1 variance_schema = test.variance/1 stddev_schema = test.stddev/1 } } ] inputs = [ \"s:sample\" ] outputs = [ \"s:mean\" \"s:variance\" \"s:stddev\" \"s:sample-count\" \"s:complete\" \"s:variance-valid\" ] }",plugin,modes[mode]);
  assert(res>0&&(size_t)res<sizeof(config));assert(spa_fgn_graph_new(config,&graph)==0);
  init_raw_buffer(&in,x,sizeof(x));init_raw_buffer(&a,mean,sizeof(mean));init_raw_buffer(&b,variance,sizeof(variance));init_raw_buffer(&c,sd,sizeof(sd));init_raw_buffer(&d,&count,sizeof(count));init_raw_buffer(&e,&complete,sizeof(complete));init_raw_buffer(&f,&valid,sizeof(valid));in.header.seq=324;
  assert(spa_fgn_graph_activate(graph)==0);assert(spa_fgn_graph_process(graph,inputs,1,outputs,6)==0);assert(mean[0]==1&&mean[3]==4&&variance[0]==0&&count==1&&complete==0&&valid==(mode==2));assert(a.header.seq==324&&f.header.seq==324);
  x[0]=3;x[1]=4;x[2]=5;x[3]=6;assert(spa_fgn_graph_process(graph,inputs,1,outputs,6)==0);assert(count==2&&valid==1&&complete==(mode==0));if(mode!=2){assert(mean[0]==2&&mean[3]==5&&variance[0]==2);}
  float previous=mean[0];x[0]=NAN;in.header.seq=325;int status=spa_fgn_graph_process(graph,inputs,1,outputs,6);if(mode==0){assert(status==0&&a.header.seq==325);}else{assert(status<0&&a.header.seq==0&&f.header.seq==0);}assert(mean[0]==previous&&count==2);
  assert(spa_fgn_graph_reset(graph)==0);x[0]=7;assert(spa_fgn_graph_process(graph,inputs,1,outputs,6)==0);assert(mean[0]==7&&count==1&&complete==0&&a.header.seq==325);
  assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);
 }
}
static void test_catalog_validity_gated_integrator(const char *plugin)
{
 struct spa_fgn_graph *graph=NULL;char config[1600];float x[1]={NAN},feedback[1]={NAN},y[1]={71},state[1]={72};uint8_t valid[2]={0,1};struct raw_test_buffer a,b,c,d,e;struct spa_buffer *inputs[]={&a.buffer,&b.buffer,&c.buffer},*outputs[]={&d.buffer,&e.buffer};
 int res=snprintf(config,sizeof(config),"{ nodes = [ { type = ndarray name = g plugin = \"%s\" label = validity-gated-closed-loop-integrator-f32 config = { state_length = 1 validity_length = 2 validity_schema = test.validity/1 } props = { gain = 0.5 pole = 0.75 anti-windup-gain = 0.25 } } ] inputs = [ \"g:residual-error\" \"g:constraint-feedback\" \"g:measurement-validity\" ] outputs = [ \"g:correction\" \"g:controller-state\" ] }",plugin);
 assert(res>0&&(size_t)res<sizeof(config));assert(spa_fgn_graph_new(config,&graph)==0);init_raw_buffer(&a,x,sizeof(x));init_raw_buffer(&b,feedback,sizeof(feedback));init_raw_buffer(&c,valid,sizeof(valid));init_raw_buffer(&d,y,sizeof(y));init_raw_buffer(&e,state,sizeof(state));a.header.seq=326;
 assert(spa_fgn_graph_activate(graph)==0);assert(spa_fgn_graph_process(graph,inputs,3,outputs,2)==SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);assert(y[0]==0&&state[0]==0&&d.header.seq==326&&e.header.seq==326);
 valid[0]=1;x[0]=4;feedback[0]=2;assert(spa_fgn_graph_process(graph,inputs,3,outputs,2)==0);assert(y[0]==1.625f&&state[0]==-0.5f);
 valid[1]=0;x[0]=NAN;feedback[0]=NAN;assert(spa_fgn_graph_process(graph,inputs,3,outputs,2)==0);assert(y[0]==1.625f&&state[0]==-0.5f);
 valid[1]=1;x[0]=8;feedback[0]=4;assert(spa_fgn_graph_process(graph,inputs,3,outputs,2)==0);assert(y[0]==4.46875f&&state[0]==0.625f);
 x[0]=NAN;a.header.seq=327;assert(spa_fgn_graph_process(graph,inputs,3,outputs,2)<0);assert(y[0]==4.46875f&&state[0]==0.625f&&d.header.seq==0&&e.header.seq==0);
 assert(spa_fgn_graph_reset(graph)==0);x[0]=4;feedback[0]=NAN;assert(spa_fgn_graph_process(graph,inputs,3,outputs,2)==0);assert(y[0]==2&&state[0]==0&&d.header.seq==327);
 assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);
}
static void test_catalog_additions_declarations(const char *plugin)
{test_catalog_pixel_validity(plugin);test_catalog_physical_offset(plugin);test_catalog_array_statistics(plugin);test_catalog_validity_gated_integrator(plugin);}
