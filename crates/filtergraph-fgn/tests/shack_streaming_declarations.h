/* SPDX-License-Identifier: MIT */
/* Native adapter checks; scientific values are qualified separately against JFG. */
static void test_shack_streaming_declarations(const char *plugin)
{
	for (unsigned kind = 0; kind < 3; kind++) {
		struct spa_fgn_graph *graph = NULL;
		char config[4096];
		const char *geometry, *ports;
		float pixels[4] = { 1, 3, 1, 3 }, values[4] = { -99, -99, -99, -99 };
		uint32_t columns[4] = { 999, 999, 999, 999 }, count[1] = { 999 };
		uint8_t active[2] = { 1, 1 };
		struct raw_test_buffer input, output, column_buffer, count_buffer, parameter;
		struct spa_buffer *inputs[] = { &input.buffer, NULL };
		struct spa_buffer *outputs[] = { &output.buffer, &column_buffer.buffer, &count_buffer.buffer };
		if (kind == 0) {
			geometry = " label = shack-hartmann-weighted-measurement-block-f32 config = {"
				" image_size = [ 4 2 ] row_block_rows = 2"
				" initial_weights = [ [ 1 1 1 1 ] [ 1 2 1 2 ] ]";
		} else if (kind == 1) {
			geometry = " label = shwfs-readout-measurement-block-f32 config = {"
				" detector_size = [ 4 2 ] region_lines = 2 block_lines = 1 region_samples = 2"
				" detector_origins = [ [ 0 0 ] [ 3 1 ] ]"
				" detector_line_steps = [ [ 1 0 ] [ -1 0 ] ]"
				" detector_sample_steps = [ [ 0 1 ] [ 0 -1 ] ]";
		} else {
			geometry = " label = shwfs-row-reconstructor-f32 config = {"
				" image_size = [ 4 2 ] row_block_rows = 2"
				" reconstructed_count = 2 initial_reconstructor = [ 1 0 0 0 0 1 0 0 ]"
				" reconstructed_schema = test.shack-stream/reconstructed/1";
		}
		ports = kind == 2 ? " outputs = [ \"stream:reconstructed\" ] }" :
			" outputs = [ \"stream:measurements\" \"stream:columns\" \"stream:count\" ] }";
		int res = snprintf(config, sizeof(config),
			"{ nodes = [ { type = ndarray name = stream plugin = \"%s\" %s"
			" subaperture_size = [ 2 2 ] subaperture_count = 2"
			" initial_subaperture_origins = [ [ 2 0 ] [ 0 0 ] ]"
			" coordinate_scale = 1 pixel_threshold = 0 flux_threshold = 0 reference_slopes = null active = null"
			" %s rate = [ 10 1 ] } } ]"
			" inputs = [ \"stream:%s\" \"stream:active\" ] %s", plugin, geometry,
			kind == 2 ? "" : "measurements_schema = test.shack-stream/measurements/1 column_indices_schema = test.shack-stream/columns/1 measurement_count_schema = test.shack-stream/count/1",
			kind == 1 ? "readout-block" : "row-block", ports);
		assert(res > 0 && (size_t)res < sizeof(config));
		assert(spa_fgn_graph_new(config, &graph) == 0);
		init_raw_buffer(&input, pixels, sizeof(pixels));
		init_raw_buffer(&output, values, kind == 2 ? 2 * sizeof(float) : sizeof(values));
		init_raw_buffer(&column_buffer, columns, sizeof(columns));
		init_raw_buffer(&count_buffer, count, sizeof(count));
		init_raw_buffer(&parameter, active, sizeof(active));
		parameter.chunk.size = 1; /* Incorrect active-mask extent. */
		input.header.seq = 90;
		input.header.offset = 0;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, kind == 2 ? 1 : 3) == 0);
		if (kind == 2) {
			assert(output.chunk.size == 0 && values[0] == -99);
		} else {
			assert(output.chunk.size == sizeof(values) && count[0] == 0);
			for (unsigned i = 0; i < 4; i++) assert(values[i] == -99 && columns[i] == 999);
		}
		assert(spa_fgn_graph_update_parameter(graph, 1, &parameter.buffer) < 0);
		input.header.offset = kind == 1 ? 1 : 2;
		input.header.flags = SPA_META_HEADER_FLAG_MARKER;
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, kind == 2 ? 1 : 3) == 0);
		assert(output.chunk.size == (kind == 2 ? 2 * sizeof(float) : sizeof(values)));
		assert(output.header.seq == 90 && output.header.offset == (kind == 2 ? 0 : input.header.offset));
		assert(!(output.header.flags & SPA_META_HEADER_FLAG_DISCONT));
		if (kind != 2) {
			assert(count[0] == 4);
			for (unsigned i = 0; i < 4; i++) assert(columns[i] == i);
		}
		assert(spa_fgn_graph_reset(graph) == 0);
		input.header.seq = 91; input.header.offset = 0; input.header.flags = 0;
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, kind == 2 ? 1 : 3) == 0);
		if (kind != 2) assert(!!(output.header.flags & SPA_META_HEADER_FLAG_DISCONT) == (kind == 1));
		/* Successful adoption abandons this partial frame; continuation cannot publish. */
		parameter.chunk.size = sizeof(active);
		assert(spa_fgn_graph_update_parameter(graph, 1, &parameter.buffer) == 0);
		input.header.offset = kind == 1 ? 1 : 2; input.header.flags = SPA_META_HEADER_FLAG_MARKER;
		res = spa_fgn_graph_process(graph, inputs, 2, outputs, kind == 2 ? 1 : 3);
		assert(res == 0 || res == SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(output.chunk.size == 0);
		input.header.seq = 92; input.header.offset = 0; input.header.flags = 0;
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, kind == 2 ? 1 : 3) == 0);
		if (kind != 2) assert(output.header.flags & SPA_META_HEADER_FLAG_DISCONT);
		input.header.offset = kind == 1 ? 1 : 2; input.header.flags = SPA_META_HEADER_FLAG_MARKER;
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, kind == 2 ? 1 : 3) == 0);
		assert(output.header.seq == 92 && (output.header.flags & SPA_META_HEADER_FLAG_MARKER));
		if (kind == 2) assert(output.header.flags & SPA_META_HEADER_FLAG_DISCONT);
		assert(spa_fgn_graph_deactivate(graph) == 0);
		spa_fgn_graph_free(graph);
	}
}
