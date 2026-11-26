/* SPDX-License-Identifier: MIT */
/* Native publication and sparse calibration for the remaining direct operators. */
static void test_shape_dense_ml_declarations(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[2200];
	float values[3] = { 1, 2, 4 }, result[2] = { 71, 71 };
	struct raw_test_buffer input, output, current_buffer, offset_buffer, inverse_buffer, parameter;
	struct spa_buffer *inputs[] = { &input.buffer, NULL, NULL };
	struct spa_buffer *outputs[] = { &output.buffer, &current_buffer.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = dense plugin = \"%s\" label = dense-vector-reconstructor-f32"
		" config = { input_count = 3 output_count = 2 initial_reconstructor = [ 1 2 3 -1 0.5 0.25 ]"
		" input_schema = org.example.vector/1 output_schema = org.example.result/1 reconstructor_schema = org.example.matrix/1 } } ]"
		" inputs = [ \"dense:ordered-input\" ] outputs = [ \"dense:output\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, values, sizeof(values)); init_raw_buffer(&output, result, sizeof(result));
	input.header.seq = 201;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(result[0] == 17 && result[1] == 1 && output.header.seq == 201);
	values[2] = NAN; input.header.seq = 202;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
	assert(result[0] == 17 && output.header.seq == 0);
	values[2] = 3.4028234663852886e38f;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
	assert(isinf(result[0]) && output.header.seq == 0);
	values[2] = 4;
	assert(spa_fgn_graph_reset(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(result[0] == 17 && result[1] == 1 && output.header.seq == 202);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = shape plugin = \"%s\" label = vdm-shape-offset-f32"
		" config = { active_vdm_count = 1 full_vdm_count = 2 physical_actuator_count = 2 } } ]"
		" inputs = [ \"shape:requested\" \"shape:shape-offset\" \"shape:pdm-to-vdm\" ]"
		" outputs = [ \"shape:output\" \"shape:current-offset\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	float requested[1] = { -0.0f }, shape[1] = { 71 }, current[2] = { 71, 71 };
	float offset[2] = { 0.5f, 1 }, inverse[4] = { 2, 3, 9, 8 };
	init_raw_buffer(&input, requested, sizeof(requested)); init_raw_buffer(&output, shape, sizeof(shape));
	init_raw_buffer(&current_buffer, current, sizeof(current)); init_raw_buffer(&offset_buffer, offset, sizeof(offset));
	init_raw_buffer(&inverse_buffer, inverse, sizeof(inverse)); input.header.seq = 211;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 2) == 0);
	assert(shape[0] == 0 && signbit(shape[0]) && current[0] == 0 && current[1] == 0);
	struct spa_fgn_parameter_update updates[] = {
		{ .input_port = 2, .buffer = &inverse_buffer.buffer },
		{ .input_port = 1, .buffer = &offset_buffer.buffer },
	};
	assert(spa_fgn_graph_set_parameters(graph, updates, 2) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 2) == SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	assert(shape[0] == 4 && current[0] == 0.5f && current[1] == 1);
	requested[0] = INFINITY; input.header.seq = 212;
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 2) < 0);
	assert(shape[0] == 4 && current[0] == 0.5f && output.header.seq == 0 && current_buffer.header.seq == 0);
	offset[1] = NAN; assert(spa_fgn_graph_update_parameter(graph, 1, &offset_buffer.buffer) < 0);
	requested[0] = 1; assert(spa_fgn_graph_reset(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 2) == 0);
	assert(shape[0] == 5 && current[0] == 0.5f && current[1] == 1 && output.header.seq == 212);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = model plugin = \"%s\" label = ml-llowfs-f32"
		" config = { model = legacy-simplechains-llowfs-v1 } } ]"
		" inputs = [ \"model:region-pixels\" \"model:model-parameters\" ] outputs = [ \"model:modes\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	float pixels[400], modes[14];
	float *parameters = calloc(240584, sizeof(float)); assert(parameters != NULL);
	for (unsigned i = 0; i < 400; i++) pixels[i] = 1 + i % 7;
	for (unsigned i = 0; i < 14; i++) { modes[i] = 71; parameters[240584-14+i] = (float)i - 3; }
	init_raw_buffer(&input, pixels, sizeof(pixels)); init_raw_buffer(&output, modes, sizeof(modes));
	init_raw_buffer(&parameter, parameters, 240584*sizeof(float));
	input.header.seq = 221; output.header.seq = 199;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) < 0);
	assert(modes[0] == 71 && output.header.seq == 0);
	assert(spa_fgn_graph_update_parameter(graph, 1, &parameter.buffer) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	for (unsigned i = 0; i < 14; i++) assert(modes[i] == (float)i - 3);
	assert(output.header.seq == 221);
	pixels[399] = NAN; input.header.seq = 222;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) < 0);
	assert(modes[0] == -3 && output.header.seq == 0);
	pixels[399] = 1; parameters[42] = NAN;
	assert(spa_fgn_graph_update_parameter(graph, 1, &parameter.buffer) < 0);
	assert(spa_fgn_graph_reset(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == 0);
	for (unsigned i = 0; i < 14; i++) assert(modes[i] == (float)i - 3);
	assert(output.header.seq == 222);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph); free(parameters);
}
