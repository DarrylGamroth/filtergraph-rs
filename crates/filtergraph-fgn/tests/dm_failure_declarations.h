/* SPDX-License-Identifier: MIT */
/* Failed GEMVs withhold metadata; staged optimization also preserves data. */
static void test_dm_failure_declarations(const char *plugin)
{
    static const struct {
        const char *label, *geometry, *input, *output, *first_parameter, *second_parameter;
    } cases[] = {
        { "controller-to-vdm-f32", "controller_command_count=2 active_vdm_count=2", "controller-command", "vdm-command", "controller-to-vdm", NULL },
        { "vdm-to-pdm-f32", "active_vdm_count=2 full_vdm_count=2 physical_actuator_count=2", "vdm-command", "requested-pdm-command", "active-to-full", "vdm-to-pdm" },
        { "pdm-feedback-to-vdm-f32", "active_vdm_count=2 full_vdm_count=2 physical_actuator_count=2", "pdm-constraint-feedback", "vdm-constraint-feedback", "full-to-active", "pdm-to-vdm" },
        { "vdm-feedback-to-controller-f32", "controller_command_count=2 active_vdm_count=2", "vdm-constraint-feedback", "controller-constraint-feedback", "vdm-to-controller", NULL },
    };
    for (unsigned c=0;c<SPA_N_ELEMENTS(cases);c++) {
        char config[1800], extra[120];
        struct spa_fgn_graph *graph=NULL;
        struct raw_test_buffer input,output,first,second;
        float values[2]={1,2},result[2]={71,71};
        float identity[4]={1,0,0,1},twice[4]={2,0,0,2};
        struct spa_buffer *inputs[]={&input.buffer,NULL,NULL},*outputs[]={&output.buffer};
        if (cases[c].second_parameter)
            snprintf(extra,sizeof(extra),"\"dm:%s\"",cases[c].second_parameter);
        else extra[0]='\0';
        int n=snprintf(config,sizeof(config),
            "{ nodes=[ { type=ndarray name=dm plugin=\"%s\" label=%s config={ %s } } ]"
            " inputs=[ \"dm:%s\" \"dm:%s\" %s ] outputs=[ \"dm:%s\" ] }",
            plugin,cases[c].label,cases[c].geometry,cases[c].input,cases[c].first_parameter,extra,cases[c].output);
        assert(n>0 && (size_t)n<sizeof(config));assert(spa_fgn_graph_new(config,&graph)==0);
        init_raw_buffer(&input,values,sizeof(values));init_raw_buffer(&output,result,sizeof(result));
        init_raw_buffer(&first,cases[c].second_parameter ? identity : twice,sizeof(twice));
        init_raw_buffer(&second,twice,sizeof(twice));input.header.seq=241;
        assert(spa_fgn_graph_activate(graph)==0);
        struct spa_fgn_parameter_update updates[]={
            {.input_port=1,.buffer=&first.buffer}, {.input_port=2,.buffer=&second.buffer},
        };
        unsigned ports=cases[c].second_parameter ? 3 : 2;
        assert(spa_fgn_graph_set_parameters(graph,updates,ports-1)==0);
        assert(spa_fgn_graph_process(graph,inputs,ports,outputs,1)==SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
        assert(result[0]==2 && result[1]==4 && output.header.seq==241);
        values[0]=NAN;input.header.seq=242;
        assert(spa_fgn_graph_process(graph,inputs,ports,outputs,1)<0);
        assert(result[0]==2 && result[1]==4 && output.header.seq==0);
        values[0]=values[1]=FLT_MAX;
        assert(spa_fgn_graph_process(graph,inputs,ports,outputs,1)<0);
        assert((!isfinite(result[0]) || !isfinite(result[1])) && output.header.seq==0);
        values[0]=1;values[1]=2;
        assert(spa_fgn_graph_process(graph,inputs,ports,outputs,1)==0);
        assert(result[0]==2 && result[1]==4 && output.header.seq==242);
        assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);
    }
    char config[1600];struct spa_fgn_graph *graph=NULL;
    int n=snprintf(config,sizeof(config),
        "{ nodes=[ { type=ndarray name=opt plugin=\"%s\" label = pdm-stroke-optimization-f32"
        " config = { actuator_count=2 null_mode_count=1 preferred_command=[0 0] command_to_null_modes=[1 1] null_modes_to_command=[1 1] } } ]"
        " inputs=[\"opt:requested\"] outputs=[\"opt:optimized\"] }",plugin);
    assert(n>0 && (size_t)n<sizeof(config));assert(spa_fgn_graph_new(config,&graph)==0);
    float values[2]={1,2},result[2]={71,71};struct raw_test_buffer input,output;
    struct spa_buffer *inputs[]={&input.buffer},*outputs[]={&output.buffer};
    init_raw_buffer(&input,values,sizeof(values));init_raw_buffer(&output,result,sizeof(result));input.header.seq=251;
    assert(spa_fgn_graph_activate(graph)==0);assert(spa_fgn_graph_process(graph,inputs,1,outputs,1)==0);
    assert(result[0]==-2 && result[1]==-1 && output.header.seq==251);
    values[0]=values[1]=FLT_MAX;input.header.seq=252;
    assert(spa_fgn_graph_process(graph,inputs,1,outputs,1)<0);
    assert(result[0]==-2 && result[1]==-1 && output.header.seq==0);
    values[0]=1;values[1]=2;assert(spa_fgn_graph_reset(graph)==0);
    assert(spa_fgn_graph_process(graph,inputs,1,outputs,1)==0);
    assert(result[0]==-2 && result[1]==-1 && output.header.seq==252);
    assert(spa_fgn_graph_deactivate(graph)==0);spa_fgn_graph_free(graph);
}
