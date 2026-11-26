/* SPDX-License-Identifier: MIT */
/* Complete-frame pupil sensing: native layouts, sparse calibration, and publication. */
static void test_pupil_signal_declarations(const char *plugin)
{
	static const struct {
		const char *declaration;
		const char *inputs;
		const char *outputs;
		unsigned image_count, signal_count;
		bool signature, divisor;
		float image[4], expected[2], replacement[2], changed[2];
		unsigned replacement_count;
	} cases[] = {
		{ "label = pyramid-image-f32 config = { image_size = [ 2 2 ] pupil_size = [ 1 1 ]"
		  " pupil_origins = [ [ 0 0 ] [ 1 0 ] [ 1 1 ] [ 0 1 ] ] normalization_policy = mean-valid-flux incidence_flux = 1  i4q_support = null  initial_reference_signal = null  initial_optical_gain = null }",
		  "\"pupil:image\" \"pupil:optical-gain\"", "\"pupil:signal\" \"pupil:normalization-divisor\"",
		  4, 2, false, true, { 8, 5, 2, 1 }, { 0.625f, 0.25f }, { 2, 3 }, { 1.25f, 0.75f }, 2 },
		{ "label = bi-o-edge-image-f32 config = { image_size = [ 2 2 ] pupil_size = [ 1 1 ]"
		  " pupil_origins = [ [ 0 0 ] [ 1 0 ] [ 1 1 ] [ 0 1 ] ] normalization_policy = mean-valid-flux incidence_flux = 1 initial_calibration_signature = 7  bi_o_edge_support = null  initial_reference_signal = null  initial_optical_gain = null }",
		  "\"pupil:image\" \"pupil:frame-calibration-signature\" \"pupil:optical-gain\"", "\"pupil:signal\" \"pupil:normalization-divisor\"",
		  4, 2, true, true, { 8, 5, 2, 1 }, { 0.625f, 0.25f }, { 2, 3 }, { 1.25f, 0.75f }, 2 },
		{ "label = zernike-pupil-signal-f32 config = { image_size = [ 1 2 ] normalization_policy = incidence-flux incidence_flux = 4 initial_calibration_signature = 7  zernike_pupil_support = null  initial_reference_pupil_image = null }",
		  "\"pupil:image\" \"pupil:frame-calibration-signature\" \"pupil:incidence-flux\"", "\"pupil:signal\" \"pupil:normalization-divisor\"",
		  2, 2, true, true, { 2, 6 }, { 0.5f, 1.5f }, { 8 }, { 0.25f, 0.75f }, 1 },
		{ "label = curvature-paired-image-f32 config = { image_size = [ 1 4 ] signal_size = [ 1 2 ] branch_size = [ 1 2 ]"
		  " positive_branch_origin = [ 0 0 ] negative_branch_origin = [ 0 2 ] initial_calibration_signature = 7  curvature_support = null  initial_reference_signal = null  initial_branch_scales = null }",
		  "\"pupil:image\" \"pupil:frame-calibration-signature\" \"pupil:branch-scales\"", "\"pupil:signal\"",
		  4, 2, true, false, { 3, 6, 1, 2 }, { 0.5f, 0.5f }, { 2, 1 }, { 5.0f/7.0f, 5.0f/7.0f }, 2 },
		{ "label = curvature-paired-channel-f32 config = { input_branch_rows = 2 input_channels = 2 signal_size = [ 1 2 ]"
		  " positive_branch_row = 0 negative_branch_row = 1 channel_offsets = [ 1 0 ] initial_calibration_signature = 7  curvature_support = null  initial_reference_signal = null  initial_branch_scales = null }",
		  "\"pupil:channels\" \"pupil:frame-calibration-signature\" \"pupil:branch-scales\"", "\"pupil:signal\"",
		  4, 2, true, false, { 3, 6, 1, 2 }, { 0.5f, 0.5f }, { 2, 1 }, { 5.0f/7.0f, 5.0f/7.0f }, 2 },
	};
	for (unsigned i = 0; i < SPA_N_ELEMENTS(cases); i++) {
		struct spa_fgn_graph *graph = NULL;
		char config[1800];
		float image[4], signal[2] = { 71, 71 }, divisor[1] = { 71 }, replacement[2];
		uint64_t signature[1] = { 7 };
		struct raw_test_buffer input, signature_input, output, divisor_output, parameter;
		struct spa_buffer *inputs[] = { &input.buffer, NULL, NULL };
		struct spa_buffer *outputs[] = { &output.buffer, &divisor_output.buffer };
		unsigned parameter_index = cases[i].signature ? 2 : 1;
		unsigned input_count = parameter_index + 1, output_count = cases[i].divisor ? 2 : 1;
		int res = snprintf(config, sizeof(config),
			"{ nodes = [ { type = ndarray name = pupil plugin = \"%s\" %s } ] inputs = [ %s ] outputs = [ %s ] }",
			plugin, cases[i].declaration, cases[i].inputs, cases[i].outputs);
		assert(res > 0 && (size_t)res < sizeof(config));
		assert(spa_fgn_graph_new(config, &graph) == 0);
		memcpy(image, cases[i].image, sizeof(image)); memcpy(replacement, cases[i].replacement, sizeof(replacement));
		init_raw_buffer(&input, image, cases[i].image_count * sizeof(float));
		init_raw_buffer(&signature_input, signature, sizeof(signature));
		init_raw_buffer(&output, signal, cases[i].signal_count * sizeof(float));
		init_raw_buffer(&divisor_output, divisor, sizeof(divisor));
		init_raw_buffer(&parameter, replacement, cases[i].replacement_count * sizeof(float));
		if (cases[i].signature) inputs[1] = &signature_input.buffer;
		input.header.seq = 151 + i; input.header.offset = 13;
		signature_input.header.seq = 999;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, input_count, outputs, output_count) == 0);
		assert(fabsf(signal[0] - cases[i].expected[0]) < 1e-6f && fabsf(signal[1] - cases[i].expected[1]) < 1e-6f);
		assert(output.header.seq == 151 + i && output.header.offset == 13);
		if (cases[i].divisor) {
			assert(divisor[0] == (i == 2 ? 4 : 16));
			assert(divisor_output.header.seq == 151 + i && divisor_output.header.offset == 13);
		}
		image[0] = NAN; input.header.seq = 201 + i;
		assert(spa_fgn_graph_process(graph, inputs, input_count, outputs, output_count) < 0);
		assert(fabsf(signal[0] - cases[i].expected[0]) < 1e-6f && output.header.seq == 0);
		assert(output.chunk.size == 0);
		if (cases[i].divisor) assert(divisor_output.header.seq == 0 && divisor_output.chunk.size == 0);
		image[0] = cases[i].image[0];
		if (cases[i].signature) {
			signature[0] = 8;
			assert(spa_fgn_graph_process(graph, inputs, input_count, outputs, output_count) < 0);
			assert(output.header.seq == 0 && output.chunk.size == 0);
			signature[0] = 7;
		}
		assert(spa_fgn_graph_process(graph, inputs, input_count, outputs, output_count) == 0);
		assert(output.header.seq == 201 + i);
		assert(spa_fgn_graph_update_parameter(graph, parameter_index, &parameter.buffer) == 0);
		assert(spa_fgn_graph_process(graph, inputs, input_count, outputs, output_count) == SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(fabsf(signal[0] - cases[i].changed[0]) < 1e-6f && fabsf(signal[1] - cases[i].changed[1]) < 1e-6f);
		if (i == 2) assert(divisor[0] == 8);
		replacement[0] = NAN;
		assert(spa_fgn_graph_update_parameter(graph, parameter_index, &parameter.buffer) < 0);
		assert(spa_fgn_graph_process(graph, inputs, input_count, outputs, output_count) == 0);
		assert(fabsf(signal[0] - cases[i].changed[0]) < 1e-6f);
		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, input_count, outputs, output_count) == 0);
		assert(fabsf(signal[0] - cases[i].changed[0]) < 1e-6f);
		assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
	}
}
