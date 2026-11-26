/* SPDX-License-Identifier: MIT */

/* Included after the graph-host test's shared buffer helpers. */
static void test_leaky_rejection_and_reset(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[2048];
	float values[2] = { 2.0f, -2.0f }, result[2] = { -123.0f, -123.0f };
	struct raw_test_buffer input, output;
	struct spa_buffer *inputs[] = { &input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = control plugin = \"%s\""
		" label = leaky-integrator-f32 config = { extent = 2 initial_state = 1.25"
		" input_schema = pipewireao/controller-residual-error-f32/1"
		" output_schema = pipewireao/controller-command-f32/1 }"
		" props = { gain = 0.5 pole = 0.75 } } ]"
		" inputs = [ \"control:input\" ] outputs = [ \"control:output\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, values, sizeof(values));
	init_raw_buffer(&output, result, sizeof(result));
	input.header.seq = 10;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) ==
		SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	assert(result[0] == 1.9375f && result[1] == -0.0625f);
	assert(output.header.seq == 10);
	values[1] = NAN;
	input.header.seq = 11;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == -EINVAL);
	assert(result[0] == 1.9375f && result[1] == -0.0625f);
	assert(output.chunk.size == 0 && output.header.seq == 0);
	values[1] = -2.0f;
	input.header.seq = 12;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(result[0] == 2.453125f && result[1] == -1.046875f);
	assert(output.chunk.size == sizeof(result) && output.header.seq == 12);
	assert(spa_fgn_graph_reset(graph) == 0);
	input.header.seq = 13;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(result[0] == 1.9375f && result[1] == -0.0625f);
	assert(output.header.seq == 13);
	assert(spa_fgn_graph_deactivate(graph) == 0);
	spa_fgn_graph_free(graph);
}
