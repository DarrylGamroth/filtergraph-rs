/* SPDX-License-Identifier: MIT */
static void test_pyramid_composite_declarations(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[2200];
	float pixels[8] = { 1, 2, 3, 4, 5, 6, 7, 8 }, reconstructed[2] = { 71, 71 };
	struct raw_test_buffer input, output;
	struct spa_buffer *inputs[] = { &input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = dense plugin = \"%s\" label = pwfs-pupil-reconstructor-f32"
		" config = { reconstructed_count = 2 selected_pixel_count = 2"
		" initial_reconstructor = [ 1 2 3 4 5 6 7 8 0 -1 0 -1 0 -1 0 -1 ] reconstructed_schema = test.pyramid/reconstructed/1 } } ]"
		" inputs = [ \"dense:reconstruction-pixels\" ] outputs = [ \"dense:reconstructed\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, pixels, sizeof(pixels)); init_raw_buffer(&output, reconstructed, sizeof(reconstructed));
	input.header.seq = 251; input.header.offset = 17;
	assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(reconstructed[0] == 204 && reconstructed[1] == -20 && output.header.seq == 251 && output.header.offset == 17);
	pixels[0] = NAN; input.header.seq = 252;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
	assert(reconstructed[0] == 204 && reconstructed[1] == -20 && output.header.seq == 0 && output.chunk.size == 0);
	pixels[0] = 1; assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0); assert(output.header.seq == 252);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	/* Current I4Q boundary is interleaved XY; matrix columns are all X then all Y. */
	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = differential plugin = \"%s\" label = pwfs-reconstructor-f32"
		" config = { reconstructed_count = 2 selected_coordinate_count = 3"
		" initial_reconstructor = [ 1 2 4 8 16 32 -1 0 0.5 0 -0.25 0 ] reconstructed_schema = test.pyramid/reconstructed/1 } } ]"
		" inputs = [ \"differential:signal\" ] outputs = [ \"differential:reconstructed\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	float signal[6] = { 1, 10, 2, 20, 3, 30 };
	init_raw_buffer(&input, signal, sizeof(signal)); init_raw_buffer(&output, reconstructed, sizeof(reconstructed));
	input.header.seq = 253; input.header.offset = 19;
	assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(reconstructed[0] == 1377 && reconstructed[1] == -4.5f && output.header.seq == 253 && output.header.offset == 19);
	signal[5] = NAN; input.header.seq = 254;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
	assert(reconstructed[0] == 1377 && reconstructed[1] == -4.5f && output.header.seq == 0 && output.chunk.size == 0);
	signal[5] = 30; assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0); assert(output.header.seq == 254);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	for (unsigned readout = 0; readout < 2; readout++) {
		float block[8] = { 2, 2, 2, 2, 2, 2, 2, 2 }, values[4] = { 71, 71, 71, 71 };
		uint32_t columns[4] = { 999, 999, 999, 999 }, count[1] = { 999 };
		struct raw_test_buffer source, measurements, column_buffer, count_buffer;
		struct spa_buffer *stream_inputs[] = { &source.buffer };
		struct spa_buffer *stream_outputs[] = { &measurements.buffer, &column_buffer.buffer, &count_buffer.buffer };
		const char *declaration = readout ?
			"label = pyramid-readout-measurement-block-f32 config = { detector_size = [ 4 4 ] region_lines = 2 block_lines = 1 region_samples = 4"
			" detector_origins = [ [ 0 0 ] [ 2 0 ] ] detector_line_steps = [ [ 1 0 ] [ 1 0 ] ] detector_sample_steps = [ [ 0 1 ] [ 0 1 ] ]" :
			"label = pyramid-measurement-block-f32 config = { image_size = [ 4 4 ] row_block_rows = 2";
		res = snprintf(config, sizeof(config),
			"{ nodes = [ { type = ndarray name = stream plugin = \"%s\" %s pupil_size = [ 1 1 ]"
			" pupil_origins = [ [ 0 0 ] [ 2 0 ] [ 2 2 ] [ 0 2 ] ] pupil_mask = null"
			" measurements_schema = test.pyramid/measurements/1 column_indices_schema = test.pyramid/columns/1 measurement_count_schema = test.pyramid/count/1"
			" rate = [ 10 1 ] } } ] inputs = [ \"stream:%s\" ] outputs = [ \"stream:measurements\" \"stream:columns\" \"stream:count\" ] }",
			plugin, declaration, readout ? "readout-block" : "row-block");
		assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
		init_raw_buffer(&source, block, sizeof(block)); init_raw_buffer(&measurements, values, sizeof(values));
		init_raw_buffer(&column_buffer, columns, sizeof(columns)); init_raw_buffer(&count_buffer, count, sizeof(count));
		assert(spa_fgn_graph_activate(graph) == 0);
		for (unsigned frame = 0; frame < 2; frame++) {
			for (unsigned ordinal = 0; ordinal < 2; ordinal++) {
				source.header.seq = 260 + frame; source.header.offset = readout ? ordinal : ordinal * 2;
				source.header.flags = ordinal == 1 ? SPA_META_HEADER_FLAG_MARKER : 0;
				for (unsigned j = 0; j < 8; j++) block[j] = frame == 0 ? 2 : 4;
				for (unsigned j = 0; j < 4; j++) { values[j] = 71; columns[j] = 999; }
				assert(spa_fgn_graph_process(graph, stream_inputs, 1, stream_outputs, 3) == 0);
				unsigned n = readout ? (ordinal == 0 ? 4 : 0) : 2;
				assert(count[0] == n && measurements.chunk.size == sizeof(values));
				assert(measurements.header.seq == 260 + frame && measurements.header.offset == source.header.offset);
				assert(column_buffer.header.seq == 260 + frame && count_buffer.header.seq == 260 + frame);
				for (unsigned j = 0; j < n; j++) assert(values[j] == (frame == 0 ? 0 : 2));
				for (unsigned j = n; j < 4; j++) assert(values[j] == 71 && columns[j] == 999);
				if (readout && ordinal == 0) for (unsigned j = 0; j < 4; j++) assert(columns[j] == j);
				if (!readout) assert(columns[0] == (ordinal == 0 ? 0 : 1) && columns[1] == (ordinal == 0 ? 3 : 2));
			}
		}
		source.header.seq = 262; source.header.offset = 0; source.header.flags = 0; block[0] = NAN;
		assert(spa_fgn_graph_process(graph, stream_inputs, 1, stream_outputs, 3) < 0);
		assert(measurements.chunk.size == 0 && column_buffer.chunk.size == 0 && count_buffer.chunk.size == 0);
		assert(measurements.header.seq == 0);
		block[0] = 4; source.header.seq = 263;
		assert(spa_fgn_graph_process(graph, stream_inputs, 1, stream_outputs, 3) == 0);
		assert(measurements.header.seq == 263 && (measurements.header.flags & SPA_META_HEADER_FLAG_DISCONT));
		for (unsigned j = 0; j < count[0]; j++) assert(values[j] == 1);
		assert(spa_fgn_graph_reset(graph) == 0); source.header.seq = 264;
		assert(spa_fgn_graph_process(graph, stream_inputs, 1, stream_outputs, 3) == 0);
		for (unsigned j = 0; j < count[0]; j++) assert(values[j] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
	}
}
