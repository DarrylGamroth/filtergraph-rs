/* SPDX-License-Identifier: MIT */

/* Included by the C graph-host test after its shared buffer helpers. */
static void test_ndarray_copy(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1024];
	float values[2] = { -0.0f, INFINITY }, copied[2] = { 1.0f, 2.0f };
	struct raw_test_buffer input, output;
	struct spa_buffer *inputs[] = { &input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = copy plugin = \"%s\""
		" label = ndarray-copy-f32 config = { extent = 2 schema = org.example.vector/1 } } ]"
		" inputs = [ \"copy:input\" ] outputs = [ \"copy:output\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, values, sizeof(values));
	init_raw_buffer(&output, copied, sizeof(copied));
	input.header.seq = 17;
	input.header.offset = 3;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(memcmp(values, copied, sizeof(values)) == 0);
	assert(output.header.seq == 17 && output.header.offset == 3);
	assert(spa_fgn_graph_deactivate(graph) == 0);
	spa_fgn_graph_free(graph);
}

static void test_pdm_engineering_offset(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1024];
	float values[2] = { 1.0f, 2.0f }, offset[2] = { 0.5f, -1.0f };
	float sum[2] = { 0.0f }, current[2] = { 0.0f };
	struct raw_test_buffer input, parameter, output, telemetry;
	struct spa_buffer *inputs[] = { &input.buffer, NULL };
	struct spa_buffer *outputs[] = { &output.buffer, &telemetry.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = offset plugin = \"%s\""
		" label = pdm-engineering-offset-f32 config = { actuator_count = 2 } } ]"
		" inputs = [ \"offset:requested\" \"offset:engineering-offset\" ]"
		" outputs = [ \"offset:requested-with-offset\" \"offset:current-offset\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, values, sizeof(values));
	init_raw_buffer(&parameter, offset, sizeof(offset));
	init_raw_buffer(&output, sum, sizeof(sum));
	init_raw_buffer(&telemetry, current, sizeof(current));
	input.header.seq = 23;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) == 0);
	assert(sum[0] == 1.0f && sum[1] == 2.0f);
	assert(current[0] == 0.0f && current[1] == 0.0f);
	assert(spa_fgn_graph_update_parameter(graph, 1, &parameter.buffer) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) ==
		SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	assert(sum[0] == 1.5f && sum[1] == 1.0f);
	assert(current[0] == 0.5f && current[1] == -1.0f);
	assert(output.header.seq == 23 && telemetry.header.seq == 23);
	assert(spa_fgn_graph_deactivate(graph) == 0);
	spa_fgn_graph_free(graph);
}

/* Same-schema vector declarations: numeric failure cannot publish data or metadata. */
static void test_vector_arithmetic(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1400];
	float a[2] = { 1.0f, 2.0f }, b[2] = { 3.0f, -1.0f };
	float sum[2] = { 71.0f, 71.0f }, current[2] = { 71.0f, 71.0f };
	struct raw_test_buffer left, right, output, telemetry;
	struct spa_buffer *inputs[] = { &left.buffer, &right.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &telemetry.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = sum plugin = \"%s\""
		" label = vector-linear-combination-f32 config = { extent = 2"
		" left_scale = 0.5 right_scale = -2.0 schema = org.example.vector/1 } } ]"
		" inputs = [ \"sum:left\" \"sum:right\" ] outputs = [ \"sum:output\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&left, a, sizeof(a)); init_raw_buffer(&right, b, sizeof(b));
	init_raw_buffer(&output, sum, sizeof(sum));
	left.header.seq = 31;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == 0);
	assert(sum[0] == -5.5f && sum[1] == 3.0f && output.header.seq == 31);
	a[1] = NAN; left.header.seq = 32;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) < 0);
	assert(sum[0] == -5.5f && sum[1] == 3.0f && output.header.seq == 0);
	a[1] = 2.0f;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == 0);
	assert(output.header.seq == 32);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = offset plugin = \"%s\""
		" label = vector-offset-f32 config = { extent = 2 schema = org.example.vector/1 } } ]"
		" inputs = [ \"offset:requested\" \"offset:offset\" ]"
		" outputs = [ \"offset:output\" \"offset:current-offset\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&telemetry, current, sizeof(current));
	inputs[1] = NULL;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_update_parameter(graph, 1, &right.buffer) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) == SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	assert(sum[0] == 4.0f && sum[1] == 1.0f && current[0] == 3.0f && current[1] == -1.0f);
	assert(output.header.seq == 32 && telemetry.header.seq == 32);
	a[1] = INFINITY; left.header.seq = 33;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) < 0);
	assert(sum[0] == 4.0f && sum[1] == 1.0f && current[0] == 3.0f && current[1] == -1.0f);
	assert(output.header.seq == 0 && telemetry.header.seq == 0);
	a[1] = 2.0f;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) == 0);
	assert(output.header.seq == 33 && telemetry.header.seq == 33);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_frame_delay_and_vector_conditioning(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1400];
	float values[2] = { 1.0f, -2.0f }, delayed[2] = { 71.0f, 71.0f };
	float feedback[2] = { 71.0f, 71.0f };
	struct raw_test_buffer input, output, telemetry;
	struct spa_buffer *inputs[] = { &input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &telemetry.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = delay plugin = \"%s\""
		" label = frame-delay-f32 config = { extent = 2 delay_frames = 1 initial_value = 0.5"
		" input_schema = org.example.vector/1 output_schema = org.example.vector/1 } } ]"
		" inputs = [ \"delay:input\" ] outputs = [ \"delay:output\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, values, sizeof(values)); init_raw_buffer(&output, delayed, sizeof(delayed));
	input.header.seq = 41;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(delayed[0] == 0.5f && delayed[1] == 0.5f && output.header.seq == 41);
	values[1] = NAN; input.header.seq = 42;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
	assert(delayed[0] == 0.5f && delayed[1] == 0.5f && output.header.seq == 0);
	values[0] = 3.0f; values[1] = 4.0f;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(delayed[0] == 1.0f && delayed[1] == -2.0f && output.header.seq == 42);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = condition plugin = \"%s\""
		" label = vector-conditioning-f32 config = { count = 2 command_limits = [ -1.0 1.0"
		" ]  slew_limit = 0.25 requested_schema = org.example.vector/1 demanded_schema = org.example.vector/1"
		" feedback_schema = org.example.vector/1  initial_current = null  quantization_origin = null  quantization_step = null  dither_seed = null } } ]"
		" inputs = [ \"condition:requested\" ] outputs = [ \"condition:demanded\" \"condition:constraint-feedback\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&telemetry, feedback, sizeof(feedback));
	values[0] = 0.75f; values[1] = -0.75f;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) == 0);
	assert(delayed[0] == 0.25f && delayed[1] == -0.25f && feedback[0] == 0.5f && feedback[1] == -0.5f);
	values[1] = INFINITY; input.header.seq = 43;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) < 0);
	assert(delayed[0] == 0.25f && delayed[1] == -0.25f && output.header.seq == 0 && telemetry.header.seq == 0);
	values[1] = -0.75f;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) == 0);
	assert(delayed[0] == 0.5f && delayed[1] == -0.5f && feedback[0] == 0.25f && feedback[1] == -0.25f);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_command_conditioning_dac_configuration(const char *plugin)
{
	static const struct {
		const char *label;
		const char *configuration;
	} declarations[] = {
		{ "pdm-command-f32", "actuator_count = 2" },
		{ "vector-conditioning-f32", "count = 2 requested_schema = org.example.vector/1"
			" demanded_schema = org.example.vector/1 feedback_schema = org.example.vector/1" },
	};
	static const struct {
		const char *configuration;
		int result;
	} cases[] = {
		{ "", 0 },
		{ "quantization_origin = null quantization_step = null dither_seed = null", 0 },
		{ "dither_seed = 18446744073709551615", 0 },
		{ "dither_seed = -1", -EINVAL },
		{ "dither_seed = 18446744073709551616", -EINVAL },
		{ "quantization_origin = 0.0", -EINVAL },
		{ "quantization_step = 0.25", -EINVAL },
		{ "quantization_origin = 0.0 quantization_step = 0.25", -EINVAL },
	};
	for (size_t d = 0; d < SPA_N_ELEMENTS(declarations); ++d) {
		for (size_t c = 0; c < SPA_N_ELEMENTS(cases); ++c) {
			struct spa_fgn_graph *graph = NULL;
			char config[1400];
			int res = snprintf(config, sizeof(config),
				"{ nodes = [ { type = ndarray name = command plugin = \"%s\""
				" label = %s config = { %s command_limits = [ -1.0 1.0 ]"
				" slew_limit = null initial_current = null %s } } ]"
				" inputs = [ \"command:requested\" ]"
				" outputs = [ \"command:demanded\" \"command:constraint-feedback\" ] }",
				plugin, declarations[d].label, declarations[d].configuration, cases[c].configuration);
			assert(res > 0 && (size_t)res < sizeof(config));
			assert(spa_fgn_graph_new(config, &graph) == cases[c].result);
			if (cases[c].result != 0)
				continue;

			float values[2] = { 0.1875f, -1.5f }, demanded[2] = { 0 }, feedback[2] = { 0 };
			struct raw_test_buffer input, output, telemetry;
			struct spa_buffer *inputs[] = { &input.buffer };
			struct spa_buffer *outputs[] = { &output.buffer, &telemetry.buffer };
			init_raw_buffer(&input, values, sizeof(values));
			init_raw_buffer(&output, demanded, sizeof(demanded));
			init_raw_buffer(&telemetry, feedback, sizeof(feedback));
			assert(spa_fgn_graph_activate(graph) == 0);
			assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) == 0);
			assert(demanded[0] == 0.1875f && demanded[1] == -1.0f);
			assert(feedback[0] == 0.0f && feedback[1] == -0.5f);
			assert(spa_fgn_graph_deactivate(graph) == 0);
			spa_fgn_graph_free(graph);
		}
	}
}

static void test_conditioning_feedback_overflow(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1400];
	float values[2] = { 1.0f, 3.0e38f }, demanded[2] = { 71.0f, 71.0f }, feedback[2] = { -17.0f, -17.0f };
	struct raw_test_buffer input, output, telemetry;
	struct spa_buffer *inputs[] = { &input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &telemetry.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = condition plugin = \"%s\""
		" label = vector-conditioning-f32 config = { count = 2 command_limits = [ -3.0e38 3.0e38"
		" ]  slew_limit = 0 initial_current = [ 0 -3.0e38 ] requested_schema = org.example.vector/1 demanded_schema = org.example.vector/1"
		" feedback_schema = org.example.vector/1  quantization_origin = null  quantization_step = null  dither_seed = null } } ]"
		" inputs = [ \"condition:requested\" ] outputs = [ \"condition:demanded\" \"condition:constraint-feedback\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, values, sizeof(values)); init_raw_buffer(&output, demanded, sizeof(demanded));
	init_raw_buffer(&telemetry, feedback, sizeof(feedback));
	input.header.seq = 51;
	output.header.seq = 40; telemetry.header.seq = 39;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) < 0);
	assert(demanded[0] == 71.0f && demanded[1] == 71.0f && feedback[0] == -17.0f && feedback[1] == -17.0f);
	assert(output.header.seq == 0 && telemetry.header.seq == 0);
	values[0] = 0; values[1] = -3.0e38f;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) == 0);
	assert(demanded[0] == 0 && demanded[1] == -3.0e38f && feedback[0] == 0 && feedback[1] == 0);
	assert(output.header.seq == 51 && telemetry.header.seq == 51);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_signed_pixel_and_biquad(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1400];
	float values[2] = { -0.5f, 2.0f }, calibrated[2] = { 71.0f, 71.0f };
	struct raw_test_buffer input, output;
	struct spa_buffer *inputs[] = { &input.buffer, NULL };
	struct spa_buffer *outputs[] = { &output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = cal plugin = \"%s\""
		" label = pixel-calibration-f32 config = { image_size = [ 1 2 ] input_schema = org.example.detector-signed/1"
		" initial_flat = [ 2 3 ] initial_background = [ 0.25 0.5 ] } } ]"
		" inputs = [ \"cal:raw\" ] outputs = [ \"cal:calibrated\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, values, sizeof(values)); init_raw_buffer(&output, calibrated, sizeof(calibrated));
	input.header.seq = 61;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(calibrated[0] == -1.5f && calibrated[1] == 4.5f && output.header.seq == 61);
	values[1] = INFINITY; input.header.seq = 62;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
	assert(calibrated[0] == -1.5f && calibrated[1] == 4.5f && output.header.seq == 0);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = biquad plugin = \"%s\""
		" label = biquad-cascade-f32 config = { extent = 2 section_count = 1 per_mode = false"
		" initial_coefficients = [ 0.5 0.25 0 -0.5 0.125 ] input_schema = org.example.vector/1 output_schema = org.example.vector/1 } } ]"
		" inputs = [ \"biquad:input\" \"biquad:coefficients\" ] outputs = [ \"biquad:output\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	values[0] = 1; values[1] = 2;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == 0);
	assert(calibrated[0] == 0.5f && calibrated[1] == 1.0f && output.header.seq == 62);
	values[1] = NAN; input.header.seq = 63;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) < 0);
	assert(calibrated[0] == 0.5f && calibrated[1] == 1.0f && output.header.seq == 0);
	values[0] = 0; values[1] = 0;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == 0);
	assert(calibrated[0] == 0.5f && calibrated[1] == 1.0f && output.header.seq == 63);
	float coefficients[5] = { 1, 0, 0, 0, 0 };
	struct raw_test_buffer parameter;
	init_raw_buffer(&parameter, coefficients, sizeof(coefficients));
	values[0] = 2; values[1] = -1; input.header.seq = 64;
	assert(spa_fgn_graph_update_parameter(graph, 1, &parameter.buffer) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	assert(calibrated[0] == 2 && calibrated[1] == -1 && output.header.seq == 64);
	coefficients[4] = NAN;
	assert(spa_fgn_graph_update_parameter(graph, 1, &parameter.buffer) < 0);
	input.header.seq = 65;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == 0);
	assert(calibrated[0] == 2 && calibrated[1] == -1 && output.header.seq == 65);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_closed_loop_integrator(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1200];
	float residual[2] = { 1.0f, -2.0f }, constraint[2] = { NAN, NAN };
	float correction[2] = { 71.0f, 71.0f }, state[2] = { 71.0f, 71.0f };
	struct raw_test_buffer input, feedback, output, telemetry;
	struct spa_buffer *inputs[] = { &input.buffer, &feedback.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &telemetry.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = integrate plugin = \"%s\""
		" label = closed-loop-integrator-f32 config = { state_length = 2 } } ]"
		" inputs = [ \"integrate:residual-error\" \"integrate:constraint-feedback\" ]"
		" outputs = [ \"integrate:correction\" \"integrate:controller-state\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, residual, sizeof(residual)); init_raw_buffer(&feedback, constraint, sizeof(constraint));
	init_raw_buffer(&output, correction, sizeof(correction)); init_raw_buffer(&telemetry, state, sizeof(state));
	input.header.seq = 71;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) == 0);
	assert(correction[0] == 1.0f && correction[1] == -2.0f && state[0] == 0.0f && state[1] == 0.0f);
	constraint[0] = 0; constraint[1] = 0; residual[1] = NAN; input.header.seq = 72;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) < 0);
	assert(correction[0] == 1.0f && correction[1] == -2.0f && output.header.seq == 0 && telemetry.header.seq == 0);
	residual[1] = 3.0f;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 2) == 0);
	assert(correction[0] == 1.0f && correction[1] == 3.0f && state[0] == 1.0f && state[1] == -2.0f);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_weighted_centroid(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1400];
	float pixels[2] = { 1.0f, 1.0f }, centroid[2] = { 71.0f, 71.0f }, flux[1] = { 71.0f };
	bool valid[1] = { false };
	struct raw_test_buffer input, output, flux_output, validity_output;
	struct spa_buffer *inputs[] = { &input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &flux_output.buffer, &validity_output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = centroid plugin = \"%s\""
		" label = weighted-center-of-gravity-f32 config = { pixel_count = 2"
		" initial_coordinates = [ [ 0 0 ] [ 2 0 ] ] initial_weights = [ 1 3 ] } } ]"
		" inputs = [ \"centroid:pixels\" ] outputs = [ \"centroid:centroid-offset\" \"centroid:flux\" \"centroid:validity\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, pixels, sizeof(pixels)); init_raw_buffer(&output, centroid, sizeof(centroid));
	init_raw_buffer(&flux_output, flux, sizeof(flux)); init_raw_buffer(&validity_output, valid, sizeof(valid));
	input.header.seq = 81;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 3) == 0);
	assert(centroid[0] == 1.5f && centroid[1] == 0.0f && flux[0] == 4.0f && valid[0]);
	assert(output.header.seq == 81 && flux_output.header.seq == 81 && validity_output.header.seq == 81);
	pixels[1] = NAN;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 3) == 0);
	assert(!valid[0] && !isfinite(flux[0]));
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_corrector_allocation(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1800];
	float command[1] = { 1.0f }, commands[2] = { 71.0f, 71.0f }, feedback[1] = { 71.0f };
	struct raw_test_buffer input, output, telemetry;
	struct spa_buffer *inputs[] = { &input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &telemetry.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = allocate plugin = \"%s\""
		" label = corrector-allocation-f32 config = { controller_command_count = 1 stage_command_counts = [ 1 1 ]"
		" update_periods = [ 1 2 ] command_projection = [ 1 1 ] realization_projection = [ 0.5 0.5 ]"
		" section_count = 1 initial_coefficients = [ 1 0 0 0 0 1 0 0 0 0 ] command_limits = [ [ -2 -2 ] [ 2 2 ] ] command_schema = org.example.vector/1  slew_limits = null } } ]"
		" inputs = [ \"allocate:controller-command\" ] outputs = [ \"allocate:commands\" \"allocate:controller-constraint-feedback\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, command, sizeof(command)); init_raw_buffer(&output, commands, sizeof(commands)); init_raw_buffer(&telemetry, feedback, sizeof(feedback));
	input.header.seq = 91;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) == 0);
	assert(commands[0] == 1.0f && commands[1] == 0.5f && feedback[0] == 0.25f);
	command[0] = NAN; input.header.seq = 92;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) < 0);
	assert(commands[0] == 1.0f && commands[1] == 0.5f && feedback[0] == 0.25f && output.header.seq == 0 && telemetry.header.seq == 0);
	command[0] = 2.0f;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) == 0);
	assert(commands[0] == 2.0f && commands[1] == 0.5f && feedback[0] == 0.75f);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 2) == 0);
	assert(commands[0] == 2.0f && commands[1] == 1.0f && feedback[0] == 0.5f);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_shack_summary_and_weighted_image(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1800];
	float slopes[4] = { 3, 4, 0, 0 }, flux[2] = { 2, 6 }, summary[8] = { 0 };
	bool valid[2] = { true, true };
	struct raw_test_buffer input, flux_input, valid_input, output;
	struct spa_buffer *inputs[] = { &input.buffer, &flux_input.buffer, &valid_input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = summary plugin = \"%s\""
		" label = shack-hartmann-summary-f32 config = { subaperture_count = 2 } } ]"
		" inputs = [ \"summary:slopes\" \"summary:flux\" \"summary:validity\" ] outputs = [ \"summary:summary\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, slopes, sizeof(slopes)); init_raw_buffer(&flux_input, flux, sizeof(flux));
	init_raw_buffer(&valid_input, valid, sizeof(valid)); init_raw_buffer(&output, summary, sizeof(summary));
	input.header.seq = 101;
	assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 1) == 0);
	assert(summary[0] == 2 && summary[1] == 2 && summary[2] == 4 && summary[6] == 1.5f && summary[7] == 2.0f && output.header.seq == 101);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = weighted plugin = \"%s\""
		" label = shack-hartmann-weighted-centroid-f32 config = { image_size = [ 1 2 ]  image_schema = pipewireao/calibrated-pixels-f32/1"
		" subaperture_size = [ 1 2 ]  subaperture_count = 1 initial_subaperture_origins = [ [ 0 0 ] ]"
		" coordinate_scale = 1 pixel_threshold = 0 flux_threshold = 0 initial_weights = [ [ 1 3 ] ]  reference_slopes = null  active = null } } ]"
		" inputs = [ \"weighted:image\" ] outputs = [ \"weighted:slopes\" \"weighted:flux\" \"weighted:validity\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	float pixels[2] = { 1, 1 }, measured[2] = { 71, 71 }, weighted_flux[1] = { 71 };
	bool weighted_valid[1] = { false };
	struct raw_test_buffer flux_output, valid_output;
	struct spa_buffer *weighted_outputs[] = { &output.buffer, &flux_output.buffer, &valid_output.buffer };
	init_raw_buffer(&input, pixels, sizeof(pixels)); init_raw_buffer(&output, measured, sizeof(measured));
	init_raw_buffer(&flux_output, weighted_flux, sizeof(weighted_flux)); init_raw_buffer(&valid_output, weighted_valid, sizeof(weighted_valid));
	input.header.seq = 102;
	assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 1, weighted_outputs, 3) == 0);
	assert(weighted_flux[0] == 4 && weighted_valid[0] && isfinite(measured[0]) && measured[1] == 0);
	assert(output.header.seq == 102 && flux_output.header.seq == 102 && valid_output.header.seq == 102);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_scc_residual_and_reconstruction(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1600];
	float difference[3] = { 1, 2, 3 }, residual[2] = { 71, 71 }, fraction[1] = { 71 };
	bool upstream[3] = { true, true, true }, selected_valid[2] = { false, false };
	uint8_t pair_status[1] = { 1 };
	struct raw_test_buffer input, validity_input, status_input, output, validity_output, fraction_output;
	struct spa_buffer *inputs[] = { &input.buffer, &validity_input.buffer, &status_input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &validity_output.buffer, &fraction_output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = residual plugin = \"%s\""
		" label = scc-pixel-residual-f32 config = { pixel_count = 3 mask = [ true false true ]"
		" initial_reference_image = [ 0.5 0 -0.25 ] minimum_valid_fraction = 0.5 } } ]"
		" inputs = [ \"residual:difference\" \"residual:difference-validity\" \"residual:pair-status\" ]"
		" outputs = [ \"residual:selected-residuals\" \"residual:selected-validity\" \"residual:valid-fraction\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, difference, sizeof(difference)); init_raw_buffer(&validity_input, upstream, sizeof(upstream));
	init_raw_buffer(&status_input, pair_status, sizeof(pair_status)); init_raw_buffer(&output, residual, sizeof(residual));
	init_raw_buffer(&validity_output, selected_valid, sizeof(selected_valid)); init_raw_buffer(&fraction_output, fraction, sizeof(fraction));
	input.header.seq = 111;
	assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 3) == 0);
	assert(residual[0] == 0.5f && residual[1] == 3.25f && selected_valid[0] && selected_valid[1] && fraction[0] == 1);
	pair_status[0] = 0; input.header.seq = 112;
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 3) < 0);
	assert(residual[0] == 0.5f && residual[1] == 3.25f && output.header.seq == 0);
	pair_status[0] = 1; upstream[2] = false;
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 3) == 0);
	assert(residual[0] == 0.5f && residual[1] == 0 && selected_valid[0] && !selected_valid[1] && fraction[0] == 0.5f);
	upstream[0] = false; input.header.seq = 113;
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 3) < 0);
	/* JFG computes complete-pair diagnostic outputs even when validity is insufficient. */
	assert(residual[0] == 0 && residual[1] == 0 && !selected_valid[0] && !selected_valid[1] && fraction[0] == 0);
	assert(output.header.seq == 0 && fraction_output.header.seq == 0);
	upstream[0] = true;
	assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 3) == 0);
	assert(residual[0] == 0.5f && residual[1] == 0 && fraction[0] == 0.5f && output.header.seq == 113);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = reconstruct plugin = \"%s\""
		" label = scc-reconstructor-f32 config = { controller_coordinate_count = 2 selected_pixel_count = 3"
		" initial_reconstructor = [ 1 2 3 -1 0.5 0.25 ] } } ]"
		" inputs = [ \"reconstruct:selected-residuals\" ] outputs = [ \"reconstruct:controller-residual-error\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	float selected[3] = { 1, 2, 4 }, error[2] = { 71, 71 };
	init_raw_buffer(&input, selected, sizeof(selected)); init_raw_buffer(&output, error, sizeof(error));
	input.header.seq = 113;
	assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(error[0] == 17 && error[1] == 1 && output.header.seq == 113);
	selected[2] = 3.4028234663852886e38f; input.header.seq = 114;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
	/* Dense reconstruction retains computed output on numerical rejection, as JFG does. */
	assert(isinf(error[0]) && isfinite(error[1]) && output.header.seq == 0);
	selected[2] = 4; input.header.seq = 115;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(error[0] == 17 && error[1] == 1 && output.header.seq == 115);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}
