/* SPDX-License-Identifier: MIT */

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <spa/buffer/meta.h>
#include <spa/filter-graph/filter-graph-ndarray.h>

#define TEST_ACTUATORS 277u

struct test_buffer {
	struct spa_buffer buffer;
	struct spa_data data;
	struct spa_chunk chunk;
	struct spa_meta meta;
	struct spa_meta_header header;
	float values[2];
};

struct raw_test_buffer {
	struct spa_buffer buffer;
	struct spa_data data;
	struct spa_chunk chunk;
	struct spa_meta meta;
	struct spa_meta_header header;
};

static void init_buffer(struct test_buffer *buffer)
{
	memset(buffer, 0, sizeof(*buffer));
	buffer->chunk.size = sizeof(buffer->values);
	buffer->chunk.stride = sizeof(buffer->values);
	buffer->data.type = SPA_DATA_MemPtr;
	buffer->data.flags = SPA_DATA_FLAG_READWRITE;
	buffer->data.fd = -1;
	buffer->data.maxsize = sizeof(buffer->values);
	buffer->data.data = buffer->values;
	buffer->data.chunk = &buffer->chunk;
	buffer->meta.type = SPA_META_Header;
	buffer->meta.size = sizeof(buffer->header);
	buffer->meta.data = &buffer->header;
	buffer->buffer.n_metas = 1;
	buffer->buffer.metas = &buffer->meta;
	buffer->buffer.n_datas = 1;
	buffer->buffer.datas = &buffer->data;
}

static void init_raw_buffer(struct raw_test_buffer *buffer,
		void *values, uint32_t bytes)
{
	memset(buffer, 0, sizeof(*buffer));
	buffer->chunk.size = bytes;
	buffer->chunk.stride = (int32_t)bytes;
	buffer->data.type = SPA_DATA_MemPtr;
	buffer->data.flags = SPA_DATA_FLAG_READWRITE;
	buffer->data.fd = -1;
	buffer->data.maxsize = bytes;
	buffer->data.data = values;
	buffer->data.chunk = &buffer->chunk;
	buffer->meta.type = SPA_META_Header;
	buffer->meta.size = sizeof(buffer->header);
	buffer->meta.data = &buffer->header;
	buffer->buffer.n_metas = 1;
	buffer->buffer.metas = &buffer->meta;
	buffer->buffer.n_datas = 1;
	buffer->buffer.datas = &buffer->data;
}

static void test_pyramid_pupil_flux(const char *plugin)
{
    struct spa_fgn_graph *graph = NULL;
    char config[2048];
    uint16_t raw[24];
    double flux[4] = { 0.0 };
    const double expected[4] = { 30.0, 48.0, 102.0, 120.0 };
    struct raw_test_buffer input_buffer, output_buffer;
    struct spa_buffer *inputs[1], *outputs[1];
    int res = snprintf(config, sizeof(config),
        "{ nodes = [ { type = ndarray name = flux plugin = \"%s\""
        " label = pyramid-pupil-flux-u16-f64 config = { image_size = [ 4 6 ]"
        " pupil_size = [ 2 3 ] pupil_origins = [ [ 0 0 ] [ 0 3 ] [ 2 0 ] [ 2 3 ] ] } } ]"
        " inputs = [ \"flux:raw\" ] outputs = [ \"flux:pupil-flux\" ] }", plugin);
    assert(res > 0 && (size_t)res < sizeof(config));
    assert(spa_fgn_graph_new(config, &graph) == 0);
    for (uint32_t i = 0; i < 24; i++) raw[i] = i + 1;
    init_raw_buffer(&input_buffer, raw, sizeof(raw));
    init_raw_buffer(&output_buffer, flux, sizeof(flux));
    input_buffer.header.seq = 123;
    input_buffer.header.pts = 456789;
    inputs[0] = &input_buffer.buffer;
    outputs[0] = &output_buffer.buffer;
    assert(spa_fgn_graph_activate(graph) == 0);
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
    for (uint32_t i = 0; i < 4; i++) assert(flux[i] == expected[i]);
    assert(output_buffer.header.seq == 123 && output_buffer.header.pts == 456789);
    input_buffer.chunk.size -= sizeof(uint16_t);
    for (uint32_t i = 0; i < 4; i++) flux[i] = 71.0;
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == -EMSGSIZE);
    for (uint32_t i = 0; i < 4; i++) assert(flux[i] == 71.0);
    assert(output_buffer.header.seq == 123);
    input_buffer.chunk.size = sizeof(raw);
    for (uint32_t i = 0; i < 24; i++) raw[i] = UINT16_MAX;
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
    for (uint32_t i = 0; i < 4; i++) assert(flux[i] == 393210.0);
    assert(spa_fgn_graph_deactivate(graph) == 0);
    spa_fgn_graph_free(graph);
}

static void test_remaining_production_declarations(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[4096];
	int res;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = excitation plugin = \"%s\""
		"   label = docrime-excitation-f32"
		" config = { amplitudes = [ 1.0 2.0 ] seed = 0 } }"
		"] inputs = [ ] outputs = [ \"excitation:excitation\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float output_values[2] = { 0.0f };
		struct raw_test_buffer output_buffer;
		struct spa_buffer *outputs[1];

		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, NULL, 0, outputs, 1) == 0);
		assert(isfinite(output_values[0]) && fabsf(output_values[0]) < 1.0f);
		assert(isfinite(output_values[1]) && fabsf(output_values[1]) < 2.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	/* Exercise the generated Shack-Hartmann partition callback on real C
	 * helper threads. Aligned outputs give two cache-line-isolated 64-subaperture
	 * partitions while preserving the ordinary declaration result exactly. */
	res = snprintf(config, sizeof(config),
		"{ workers = { helpers = 3 } nodes = ["
		" { type = ndarray name = measure plugin = \"%s\""
		"   label = shack-hartmann-image-f32"
		" config = { image_size = [ 8 32 ] "
		"              subaperture_size = [ 1 2 ] "
		"              subaperture_count = 128"
		"              image_schema = pipewireao/calibrated-pixels-f32/1"
		"              coordinate_scale = 1.0"
		"              pixel_threshold = 0.0 flux_threshold = 0.0  initial_subaperture_origins = null  reference_slopes = null  active = null } }"
		"] inputs = [ \"measure:image\" \"measure:subaperture-origins\""
		"             \"measure:coordinates\" \"measure:reference-slopes\""
		"             \"measure:thresholds\" \"measure:active\" ]"
		" outputs = [ \"measure:slopes\" \"measure:flux\""
		"             \"measure:validity\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float *image_values, *slope_values, *flux_values;
		uint8_t *validity_values;
		struct raw_test_buffer image_buffer, slope_buffer, flux_buffer;
		struct raw_test_buffer validity_buffer;
		struct spa_buffer *inputs[6];
		struct spa_buffer *outputs[3];

		assert(posix_memalign((void **)&image_values, SPA_CACHE_LINE_SIZE,
				256 * sizeof(float)) == 0);
		assert(posix_memalign((void **)&slope_values, SPA_CACHE_LINE_SIZE,
				256 * sizeof(float)) == 0);
		assert(posix_memalign((void **)&flux_values, SPA_CACHE_LINE_SIZE,
				128 * sizeof(float)) == 0);
		assert(posix_memalign((void **)&validity_values, SPA_CACHE_LINE_SIZE,
				128 * sizeof(uint8_t)) == 0);
		for (size_t i = 0; i < 128; i++) {
			image_values[2 * i] = 1.0f;
			image_values[2 * i + 1] = 3.0f;
		}
		memset(slope_values, 0, 256 * sizeof(float));
		memset(flux_values, 0, 128 * sizeof(float));
		memset(validity_values, 0, 128 * sizeof(uint8_t));
		init_raw_buffer(&image_buffer, image_values, 256 * sizeof(float));
		init_raw_buffer(&slope_buffer, slope_values, 256 * sizeof(float));
		init_raw_buffer(&flux_buffer, flux_values, 128 * sizeof(float));
		init_raw_buffer(&validity_buffer, validity_values,
				128 * sizeof(uint8_t));
		inputs[0] = &image_buffer.buffer;
		for (size_t i = 1; i < 6; i++)
			inputs[i] = NULL;
		outputs[0] = &slope_buffer.buffer;
		outputs[1] = &flux_buffer.buffer;
		outputs[2] = &validity_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		for (size_t iteration = 0; iteration < 32; iteration++)
			assert(spa_fgn_graph_process(graph, inputs, 6,
					outputs, 3) == 0);
		for (size_t i = 0; i < 128; i++) {
			assert(slope_values[2 * i] == 0.25f);
			assert(slope_values[2 * i + 1] == 0.0f);
			assert(flux_values[i] == 4.0f);
			assert(validity_values[i] == 1);
		}
		assert(spa_fgn_graph_deactivate(graph) == 0);
		free(validity_values);
		free(flux_values);
		free(slope_values);
		free(image_values);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = invalid-profile plugin = \"%s\""
		"   label = pixel-calibration-u16-f32"
		"   config = { image_size = [ 1 3 ] initial_flat = null initial_background = null"
		"              profiles = { missing = detector-profile-id } } }"
		"] inputs = [ \"invalid-profile:raw\""
		"             \"invalid-profile:flat\""
		"             \"invalid-profile:background\" ]"
		" outputs = [ \"invalid-profile:calibrated\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == -EINVAL);
	assert(graph == NULL);

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = excitation plugin = \"%s\""
		"   label = docrime-binary-excitation-f32"
		" config = { amplitudes = [ 1.0 2.0 ] seed = 0 } }"
		"] inputs = [ ] outputs = [ \"excitation:excitation\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float output_values[2] = { 0.0f };
		struct raw_test_buffer output_buffer;
		struct spa_buffer *outputs[1];

		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, NULL, 0, outputs, 1) == 0);
		assert(fabsf(output_values[0]) == 1.0f);
		assert(fabsf(output_values[1]) == 2.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = select plugin = \"%s\""
		"   label = pixel-selection-f32"
		" config = { image_size = [ 2 2 ] "
		"              mask = [ true false true false ]"
		"              input_schema = test.image/1"
		"              output_schema = test.selected-pixels/1 } }"
		"] inputs = [ \"select:image\" ]"
		" outputs = [ \"select:selected-pixels\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float input_values[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
		float output_values[2] = { 0.0f };
		struct raw_test_buffer input_buffer, output_buffer;
		struct spa_buffer *inputs[1], *outputs[1];

		init_raw_buffer(&input_buffer, input_values, sizeof(input_values));
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		inputs[0] = &input_buffer.buffer;
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(output_values[0] == 1.0f && output_values[1] == 3.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = regions plugin = \"%s\""
		"   label = region-extraction-f32"
		" config = { image_size = [ 2 2 ] "
		"              region_size = [ 1 2 ] "
		"              origins = [ [ 0 0 ] [ 1 0 ] ] } }"
		"] inputs = [ \"regions:calibrated\" ]"
		" outputs = [ \"regions:regions\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float input_values[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
		float output_values[4] = { 0.0f };
		struct raw_test_buffer input_buffer, output_buffer;
		struct spa_buffer *inputs[1], *outputs[1];

		init_raw_buffer(&input_buffer, input_values, sizeof(input_values));
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		inputs[0] = &input_buffer.buffer;
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(memcmp(input_values, output_values, sizeof(input_values)) == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = response plugin = \"%s\""
		"   label = scc-push-pull-response-f32"
		" config = { pixel_count = 2 } }"
		"] inputs = [ \"response:push-fringed\""
		"             \"response:push-unfringed\""
		"             \"response:pull-fringed\""
		"             \"response:pull-unfringed\" ]"
		" outputs = [ \"response:response\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float push_fringed[2] = { 10.0f, 5.0f };
		float push_unfringed[2] = { 4.0f, 7.0f };
		float pull_fringed[2] = { 3.0f, -1.0f };
		float pull_unfringed[2] = { 2.0f, 2.0f };
		float output_values[2] = { 0.0f };
		struct raw_test_buffer input_buffers[4], output_buffer;
		struct spa_buffer *inputs[4], *outputs[1];

		init_raw_buffer(&input_buffers[0], push_fringed,
				sizeof(push_fringed));
		init_raw_buffer(&input_buffers[1], push_unfringed,
				sizeof(push_unfringed));
		init_raw_buffer(&input_buffers[2], pull_fringed,
				sizeof(pull_fringed));
		init_raw_buffer(&input_buffers[3], pull_unfringed,
				sizeof(pull_unfringed));
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		for (size_t i = 0; i < 4; i++)
			inputs[i] = &input_buffers[i].buffer;
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 4, outputs, 1) == 0);
		assert(output_values[0] == 5.0f && output_values[1] == 1.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = power plugin = \"%s\""
		"   label = pdm-command-power-limit-f32"
		" config = { actuator_count = 2 max_command_power = 4.0 } }"
		"] inputs = [ \"power:requested\" ]"
		" outputs = [ \"power:power-limited\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float input_values[2] = { 3.0f, 4.0f };
		float output_values[2] = { 0.0f };
		struct raw_test_buffer input_buffer, output_buffer;
		struct spa_buffer *inputs[1], *outputs[1];

		init_raw_buffer(&input_buffer, input_values, sizeof(input_values));
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		inputs[0] = &input_buffer.buffer;
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(fabsf(output_values[0] - 1.2f) < 1.0e-6f);
		assert(fabsf(output_values[1] - 1.6f) < 1.0e-6f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = stroke-limit plugin = \"%s\""
		"   label = inter-actuator-stroke-limit-f32"
		" config = { actuator_count = 3"
		"              edges = [ [ 0 1 ] [ 1 2 ] ]"
		"              max_differences = [ 1.0 0.25 ] } }"
		"] inputs = [ \"stroke-limit:requested\" ]"
		" outputs = [ \"stroke-limit:stroke-limited\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float input_values[3] = { 2.0f, -2.0f, 1.0f };
		float output_values[3] = { 0.0f };
		struct raw_test_buffer input_buffer, output_buffer;
		struct spa_buffer *inputs[1], *outputs[1];

		init_raw_buffer(&input_buffer, input_values, sizeof(input_values));
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		inputs[0] = &input_buffer.buffer;
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(fabsf(output_values[0] - 1.0f / 6.0f) < 1.0e-6f);
		assert(fabsf(output_values[1] + 1.0f / 6.0f) < 1.0e-6f);
		assert(fabsf(output_values[2] - 1.0f / 12.0f) < 1.0e-6f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = optimize plugin = \"%s\""
		"   label = pdm-stroke-optimization-f32"
		" config = { actuator_count = 3 null_mode_count = 1"
		"              preferred_command = [ 0.0 0.0 0.0 ]"
		"              command_to_null_modes ="
		"                  [ 0.33333334 0.33333334 0.33333334 ]"
		"              null_modes_to_command = [ 1.0 1.0 1.0 ] } }"
		"] inputs = [ \"optimize:requested\" ]"
		" outputs = [ \"optimize:optimized\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float input_values[3] = { 4.0f, 1.0f, -2.0f };
		float output_values[3] = { 0.0f };
		struct raw_test_buffer input_buffer, output_buffer;
		struct spa_buffer *inputs[1], *outputs[1];

		init_raw_buffer(&input_buffer, input_values, sizeof(input_values));
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		inputs[0] = &input_buffer.buffer;
		outputs[0] = &output_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(fabsf(output_values[0] - 3.0f) < 1.0e-6f);
		assert(fabsf(output_values[1]) < 1.0e-6f);
		assert(fabsf(output_values[2] + 3.0f) < 1.0e-6f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_row_reconstructor(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[8192];
	int res;

	res = snprintf(config, sizeof(config),
		"{ workers = { helpers = 3 } nodes = ["
		""
		" { type = ndarray name = row-measure plugin = \"%s\" label = "
		"shack-hartmann-measurement-block-f32 config = { image_size = [ 6 6 ] "
		" row_block_rows = 1 subaperture_size = [ 2 2 ]  "
		"subaperture_count = 4 initial_subaperture_origins = [ [ 0 0 ] [ 0 3 ] [ 3 0 ] [ "
		"3 3 ] ] coordinate_scale = 1.0 pixel_threshold = 0.0 flux_threshold = 0.0 "
		"measurements_schema = org.example.ao.measurement-block/1 column_indices_schema "
		"= org.example.ao.reconstructor-column-indices/1 measurement_count_schema = "
		"org.example.ao.measurement-block-count/1 rate = [ 10 1 ]  reference_slopes = null  active = null } }"
		" { type = ndarray name = row-reconstruct plugin = \"%s\" label = "
		"incremental-dense-reconstructor-f32 config = { measurement_count = 8 "
		"measurement_block_capacity = 8 blocks_per_sample = 6 reconstructed_count = 64  "
		"measurements_schema = org.example.ao.measurement-block/1 column_indices_schema "
		"= org.example.ao.reconstructor-column-indices/1 measurement_count_schema = "
		"org.example.ao.measurement-block-count/1 reconstructor_schema = "
		"pipewireao/shwfs-reconstructor-f32/1 reconstructed_schema = "
		"pipewireao/controller-residual-error-f32/1 rate = [ 10 1 ]  initial_reconstructor = null } }"
		" ] links = ["
		""
		" { output = \"row-measure:measurements\" input = \"row-reconstruct:measurements\" }"
		" { output = \"row-measure:columns\" input = \"row-reconstruct:columns\" }"
		" { output = \"row-measure:count\" input = \"row-reconstruct:count\" }"
		" ] inputs = ["
		" \"row-measure:row-block\""
		" \"row-measure:subaperture-origins\" \"row-measure:coordinates\" "
		"\"row-measure:reference-slopes\" \"row-measure:thresholds\" \"row-measure:active\" "
		"\"row-reconstruct:reconstructor\" ] outputs = [  \"row-reconstruct:reconstructed\" "
		"\"row-measure:measurements\" \"row-measure:columns\" \"row-measure:count\" ] }",
		plugin, plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 7);
	assert(spa_fgn_graph_get_n_outputs(graph) == 4);
	{
		float row_values[6];
		float output_values[64];
		float measurement_values[8];
		uint32_t column_values[8], count_value;
		struct raw_test_buffer row_buffer, output_buffer, measurement_buffer, column_buffer, count_buffer;
		struct spa_buffer *inputs[7] = { NULL };
		struct spa_buffer *outputs[4];

		init_raw_buffer(&row_buffer, row_values, sizeof(row_values));
		for (size_t i = 0; i < SPA_N_ELEMENTS(output_values); i++)
			output_values[i] = -7.0f;
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		inputs[0] = &row_buffer.buffer;
		outputs[0] = &output_buffer.buffer;
		init_raw_buffer(&measurement_buffer, measurement_values, sizeof(measurement_values));
		init_raw_buffer(&column_buffer, column_values, sizeof(column_values));
		init_raw_buffer(&count_buffer, &count_value, sizeof(count_value));
		outputs[1] = &measurement_buffer.buffer;
		outputs[2] = &column_buffer.buffer;
		outputs[3] = &count_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		for (uint32_t row = 0; row < 6; row++) {
			for (uint32_t column = 0; column < 6; column++)
				row_values[column] = 1.0f + 0.2f * row +
					0.1f * column;
			row_buffer.header.seq = 91;
			row_buffer.header.offset = row;
			row_buffer.header.pts = row == 5 ? 456789 : SPA_TIME_INVALID;
			row_buffer.header.flags = row == 5 ?
				SPA_META_HEADER_FLAG_MARKER : 0;
			assert(spa_fgn_graph_process(graph, inputs, 7,
					outputs, 4) == 0);
			assert(measurement_buffer.chunk.size == sizeof(measurement_values));
			assert(column_buffer.chunk.size == sizeof(column_values));
			assert(count_buffer.chunk.size == sizeof(count_value));
			assert(count_value == (row == 1 || row == 4 ? 4u : 0u));
			for (uint32_t port = 1; port < 4; port++) {
				const struct spa_meta_header *header = outputs[port]->metas[0].data;
				assert(header->seq == 91 && header->offset == row);
				assert(header->flags == row_buffer.header.flags);
				assert(header->pts == row_buffer.header.pts);
			}
			if (row < 5) {
				assert(output_buffer.chunk.size == 0);
				for (size_t i = 0; i < SPA_N_ELEMENTS(output_values); i++)
					assert(output_values[i] == -7.0f);
			}
		}
		assert(output_buffer.chunk.size == sizeof(output_values));
		for (size_t i = 0; i < SPA_N_ELEMENTS(output_values); i++)
			assert(output_values[i] == 0.0f);
		assert(output_buffer.header.seq == 91);
		assert(output_buffer.header.offset == 0);
		assert(output_buffer.header.pts == 456789);
		assert(output_buffer.header.flags == SPA_META_HEADER_FLAG_MARKER);
		/* Missing first, middle or terminal blocks cannot reuse a previously
		 * published triplet. Each next complete frame must recover with DISCONT. */
		for (uint32_t lost = 0; lost < 7; lost++) {
			for (uint32_t pass = 0; pass < 2; pass++) {
				for (uint32_t row = 0; row < 6; row++) {
					if (pass == 0 && row == lost)
						continue;
					row_buffer.header.seq = 100 + lost * 2 + pass;
					row_buffer.header.offset = row;
					row_buffer.header.pts = SPA_TIME_INVALID;
					row_buffer.header.flags = row == 5 ? SPA_META_HEADER_FLAG_MARKER : 0;
					if (pass == 0 && lost == 6 && row == 2)
						row_buffer.header.flags |= SPA_META_HEADER_FLAG_CORRUPTED;
					assert(spa_fgn_graph_process(graph, inputs, 7, outputs, 4) == 0);
					if (pass == 0 || row < 5)
						assert(output_buffer.chunk.size == 0);
					else {
						assert(output_buffer.chunk.size == sizeof(output_values));
						assert(output_buffer.header.seq == row_buffer.header.seq);
						assert(output_buffer.header.flags & SPA_META_HEADER_FLAG_DISCONT);
					}
				}
			}
		}
		/* Independent sensing/reconstruction plan adoption abandons the partial
		 * frame; the continuation must not publish old/new calibration mixtures. */
		for (uint32_t selected = 0; selected < 2; selected++) {
			float replacement[64 * 8] = { 0.0f };
			struct raw_test_buffer parameter_buffer;
			uint32_t parameter_port = selected == 0 ? 3u : 6u;
			init_raw_buffer(&parameter_buffer, replacement,
				selected == 0 ? 8u * sizeof(float) : sizeof(replacement));
			parameter_buffer.header.seq = 1 + selected;
			for (uint32_t row = 0; row < 6; row++) {
				if (row == 2)
					assert(spa_fgn_graph_update_parameter(graph, parameter_port,
						&parameter_buffer.buffer) == 0);
				row_buffer.header.seq = 200 + selected * 2;
				row_buffer.header.offset = row;
				row_buffer.header.flags = row == 5 ? SPA_META_HEADER_FLAG_MARKER : 0;
				res = spa_fgn_graph_process(graph, inputs, 7, outputs, 4);
				assert(res == (row == 2 ? 1 : 0));
				assert(output_buffer.chunk.size == 0);
			}
			for (uint32_t row = 0; row < 6; row++) {
				row_buffer.header.seq = 201 + selected * 2;
				row_buffer.header.offset = row;
				row_buffer.header.flags = row == 5 ? SPA_META_HEADER_FLAG_MARKER : 0;
				assert(spa_fgn_graph_process(graph, inputs, 7, outputs, 4) == 0);
				assert((output_buffer.chunk.size != 0) == (row == 5));
			}
			assert(output_buffer.header.flags & SPA_META_HEADER_FLAG_DISCONT);
		}
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_raw_row_front_end(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	const struct spa_fgn_format *format;
	char config[8192];
	int res;

	res = snprintf(config, sizeof(config),
		"{ workers = { helpers = 0 } nodes = ["
		" { type = ndarray name = row-calibrate plugin = \"%s\" label = "
		"pixel-calibration-row-u16-f32 config = { image_size = [ 6 6 ]  "
		"row_block_rows = 2 rate = [ 10 1 ]  initial_flat = null  initial_background = null } }"
		" { type = ndarray name = row-measure plugin = \"%s\" label = "
		"shack-hartmann-measurement-block-f32 config = { image_size = [ 6 6 ] "
		" row_block_rows = 2 subaperture_size = [ 2 2 ]  "
		"subaperture_count = 4 initial_subaperture_origins = [ [ 0 0 ] [ 0 3 ] [ 3 0 ] [ "
		"3 3 ] ] coordinate_scale = 1.0 pixel_threshold = 0.0 flux_threshold = 0.0 "
		"measurements_schema = org.example.ao.measurement-block/1 column_indices_schema "
		"= org.example.ao.reconstructor-column-indices/1 measurement_count_schema = "
		"org.example.ao.measurement-block-count/1 rate = [ 10 1 ]  reference_slopes = null  active = null } }"
		" { type = ndarray name = row-reconstruct plugin = \"%s\" label = "
		"incremental-dense-reconstructor-f32 config = { measurement_count = 8 "
		"measurement_block_capacity = 8 blocks_per_sample = 3 reconstructed_count = 3 "
		"initial_reconstructor = [ 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 "
		"0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 ] "
		"measurements_schema = org.example.ao.measurement-block/1 column_indices_schema "
		"= org.example.ao.reconstructor-column-indices/1 measurement_count_schema = "
		"org.example.ao.measurement-block-count/1 reconstructor_schema = "
		"pipewireao/shwfs-reconstructor-f32/1 reconstructed_schema = "
		"pipewireao/controller-residual-error-f32/1 rate = [ 10 1 ] } }"
		" ] links = ["
		" { output = \"row-calibrate:calibrated\" input = \"row-measure:row-block\" }"
		" { output = \"row-measure:measurements\" input = \"row-reconstruct:measurements\" }"
		" { output = \"row-measure:columns\" input = \"row-reconstruct:columns\" }"
		" { output = \"row-measure:count\" input = \"row-reconstruct:count\" }"
		" ] inputs = ["
		" \"row-calibrate:raw\" \"row-calibrate:flat\" \"row-calibrate:background\""
		" \"row-measure:subaperture-origins\" \"row-measure:coordinates\" "
		"\"row-measure:reference-slopes\" \"row-measure:thresholds\" \"row-measure:active\" "
		"\"row-reconstruct:reconstructor\" ] outputs = [ \"row-calibrate:calibrated\" "
		"\"row-reconstruct:reconstructed\" ] }",
		plugin, plugin, plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "raw row front-end graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 9);
	assert(spa_fgn_graph_get_n_outputs(graph) == 2);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_U16_LE &&
			format->n_dimensions == 2 && format->shape[0] == 2 &&
			format->shape[1] == 6 && format->rate_num == 30 &&
			format->rate_denom == 1);
	assert(strcmp(format->schema,
			"pipewireao/raw-pixel-row-block-u16/1") == 0);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			1, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 1 && format->shape[0] == 3 &&
			format->rate_num == 10 && format->rate_denom == 1);

	{
		uint16_t raw_values[12];
		float replacement_flat[36];
		float calibrated_values[12] = { 0.0f };
		float reconstructed_values[3] = { -7.0f, -7.0f, -7.0f };
		struct raw_test_buffer raw_buffer, flat_buffer, calibrated_buffer;
		struct raw_test_buffer reconstructed_buffer;
		struct spa_buffer *inputs[9] = { NULL };
		struct spa_buffer *outputs[2];

		init_raw_buffer(&raw_buffer, raw_values, sizeof(raw_values));
		for (uint32_t pixel = 0; pixel < 36; pixel++)
			replacement_flat[pixel] = 2.0f;
		init_raw_buffer(&flat_buffer, replacement_flat,
				sizeof(replacement_flat));
		init_raw_buffer(&calibrated_buffer, calibrated_values,
				sizeof(calibrated_values));
		init_raw_buffer(&reconstructed_buffer, reconstructed_values,
				sizeof(reconstructed_values));
		inputs[0] = &raw_buffer.buffer;
		outputs[0] = &calibrated_buffer.buffer;
		outputs[1] = &reconstructed_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);

		for (uint32_t local = 0; local < 12; local++)
			raw_values[local] = 10u + local;
		raw_buffer.header.seq = 121;
		raw_buffer.header.offset = 0;
		raw_buffer.header.pts = SPA_TIME_INVALID;
		raw_buffer.header.flags = 0;
		assert(spa_fgn_graph_process(graph, inputs, 9, outputs, 2) == 0);
		for (uint32_t local = 0; local < 12; local++)
			assert(calibrated_values[local] == (float)raw_values[local]);
		assert(calibrated_buffer.header.seq == 121);
		assert(calibrated_buffer.header.offset == 0);
		assert(reconstructed_buffer.chunk.size == 0);
		assert(reconstructed_values[0] == -7.0f &&
				reconstructed_values[1] == -7.0f &&
				reconstructed_values[2] == -7.0f);

		assert(spa_fgn_graph_update_parameter(graph, 1,
				&flat_buffer.buffer) == 0);
		for (uint32_t first_row = 2; first_row < 6; first_row += 2) {
			for (uint32_t local = 0; local < 12; local++)
				raw_values[local] = 10u + first_row * 6u + local;
			raw_buffer.header.seq = 121;
			raw_buffer.header.offset = first_row;
			raw_buffer.header.pts = first_row == 4 ? 987654 :
				SPA_TIME_INVALID;
			raw_buffer.header.flags = first_row == 4 ?
				SPA_META_HEADER_FLAG_MARKER : 0;
			assert(spa_fgn_graph_process(graph, inputs, 9, outputs, 2) ==
					(first_row == 2 ?
					 SPA_FGN_PROCESS_RESULT_PROPS_CHANGED : 0));
			for (uint32_t local = 0; local < 12; local++)
				assert(calibrated_values[local] ==
						2.0f * (float)raw_values[local]);
			assert(calibrated_buffer.header.seq == 121);
			assert(calibrated_buffer.header.offset == first_row);
			if (first_row == 2)
				assert((calibrated_buffer.header.flags &
						SPA_META_HEADER_FLAG_DISCONT) != 0);
			assert(reconstructed_buffer.chunk.size == 0);
			assert(reconstructed_values[0] == -7.0f &&
					reconstructed_values[1] == -7.0f &&
					reconstructed_values[2] == -7.0f);
		}

		for (uint32_t first_row = 0; first_row < 6; first_row += 2) {
			for (uint32_t local = 0; local < 12; local++)
				raw_values[local] = 100u + first_row * 6u + local;
			raw_buffer.header.seq = 122;
			raw_buffer.header.offset = first_row;
			raw_buffer.header.pts = first_row == 4 ? 987654 :
				SPA_TIME_INVALID;
			raw_buffer.header.flags = first_row == 4 ?
				SPA_META_HEADER_FLAG_MARKER : 0;
			assert(spa_fgn_graph_process(graph, inputs, 9,
					outputs, 2) == 0);
			for (uint32_t local = 0; local < 12; local++)
				assert(calibrated_values[local] ==
						2.0f * (float)raw_values[local]);
			if (first_row < 4)
				assert(reconstructed_buffer.chunk.size == 0);
		}
		assert(reconstructed_buffer.chunk.size == sizeof(reconstructed_values));
		assert(isfinite(reconstructed_values[0]) &&
				isfinite(reconstructed_values[1]) &&
				isfinite(reconstructed_values[2]));
		assert(reconstructed_buffer.header.seq == 122);
		assert(reconstructed_buffer.header.offset == 0);
		assert(reconstructed_buffer.header.pts == 987654);
		assert(reconstructed_buffer.header.flags ==
				(SPA_META_HEADER_FLAG_MARKER |
				 SPA_META_HEADER_FLAG_DISCONT));
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_raw_row_pwfs_normalization(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[4096];
	int res;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = normalize plugin = \"%s\""
		"   label = pyramid-pixel-row-f32"
		" config = { image_size = [ 8 8 ] "
		"              row_block_rows = 2 pupil_size = [ 2 2 ] "
		"              pupil_origins ="
		"                  [ [ 6 0 ] [ 0 0 ] [ 6 4 ] [ 0 4 ] ]"
		"              pupil_mask = [ true false true true ]"
		"              rate = [ 10 1 ] } }"
		"] inputs = [ \"normalize:row-block\""
		"             \"normalize:pupil-origins\""
		"             \"normalize:pupil-mask\" ]"
		" outputs = [ \"normalize:reconstruction-pixels\""
		"             \"normalize:mean-pupil-intensity\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float row_values[16];
		float normalized_values[12];
		float mean_value[1] = { -1.0f };
		struct raw_test_buffer row_buffer, normalized_buffer, mean_buffer;
		struct spa_buffer *inputs[3] = { NULL };
		struct spa_buffer *outputs[2];

		for (uint32_t pixel = 0; pixel < 12; pixel++)
			normalized_values[pixel] = -1.0f;
		init_raw_buffer(&row_buffer, row_values, sizeof(row_values));
		init_raw_buffer(&normalized_buffer, normalized_values,
				sizeof(normalized_values));
		init_raw_buffer(&mean_buffer, mean_value, sizeof(mean_value));
		inputs[0] = &row_buffer.buffer;
		outputs[0] = &normalized_buffer.buffer;
		outputs[1] = &mean_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		for (uint32_t first_row = 0; first_row < 8; first_row += 2) {
			for (uint32_t local = 0; local < 16; local++)
				row_values[local] = 1.0f + (float)(first_row * 8 + local);
			row_buffer.header.seq = 1;
			row_buffer.header.offset = first_row;
			row_buffer.header.flags = first_row == 6 ?
				SPA_META_HEADER_FLAG_MARKER : 0;
			assert(spa_fgn_graph_process(graph, inputs, 3,
					outputs, 2) == 0);
			if (first_row < 6) {
				assert(normalized_buffer.chunk.size == 0);
				assert(mean_buffer.chunk.size == 0);
			}
		}
		assert(normalized_buffer.chunk.size == sizeof(normalized_values));
		assert(mean_buffer.chunk.size == sizeof(mean_value));
		assert(mean_value[0] == 31.5f);
		for (uint32_t pixel = 0; pixel < 12; pixel++)
			assert(normalized_values[pixel] == 0.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_raw_row_pwfs_front_end(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	const struct spa_fgn_format *format;
	char config[8192];
	int res;

	res = snprintf(config, sizeof(config),
		"{ workers = { helpers = 3 } nodes = ["
		" { type = ndarray name = row-calibrate plugin = \"%s\""
		"   label = pixel-calibration-row-u16-f32"
		" config = { image_size = [ 8 8 ] "
		"              row_block_rows = 2 rate = [ 10 1 ]  initial_flat = null  initial_background = null } }"
		" { type = ndarray name = row-pyramid plugin = \"%s\""
		"   label = pwfs-row-reconstructor-f32"
		" config = { image_size = [ 8 8 ] "
		"              row_block_rows = 2 pupil_size = [ 2 2 ] "
		"              pupil_origins ="
		"                  [ [ 6 0 ] [ 0 0 ] [ 6 4 ] [ 0 4 ] ]"
		"              pupil_mask = [ true false true true ]"
		"              reconstructed_count = 64"
		"              reconstructed_schema = test.pwfs-reconstructed/1"
		"              rate = [ 10 1 ]  initial_reconstructor = null } }"
		"] links = ["
		" { output = \"row-calibrate:calibrated\""
		"   input = \"row-pyramid:row-block\" }"
		"] inputs = [ \"row-calibrate:raw\""
		"             \"row-calibrate:flat\""
		"             \"row-calibrate:background\""
		"             \"row-pyramid:pupil-origins\""
		"             \"row-pyramid:pupil-mask\""
		"             \"row-pyramid:reconstructor\" ]"
		" outputs = [ \"row-pyramid:reconstruction-pixels\""
		"             \"row-pyramid:mean-pupil-intensity\""
		"             \"row-pyramid:reconstructed\" ] }",
		plugin, plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "raw row PWFS graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 6);
	assert(spa_fgn_graph_get_n_outputs(graph) == 3);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_U16_LE &&
			format->n_dimensions == 2 && format->shape[0] == 2 &&
			format->shape[1] == 8 && format->rate_num == 40 &&
			format->rate_denom == 1);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 2 && format->shape[0] == 4 &&
			format->shape[1] == 3 && format->rate_num == 10 &&
			format->rate_denom == 1);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			2, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 1 && format->shape[0] == 64 &&
			format->rate_num == 10 && format->rate_denom == 1);

	{
		static const uint32_t selected[] = {
			49, 57, 58, 1, 9, 10, 53, 61, 62, 5, 13, 14,
		};
		uint16_t raw_values[16];
		float reconstructor_values[64 * 12] = { 0.0f };
		float reconstruction_values[12] = { -1.0f };
		float mean_value[1] = { -1.0f };
		float reconstructed_values[64];
		struct raw_test_buffer raw_buffer, reconstructor_buffer;
		struct raw_test_buffer reconstruction_buffer, mean_buffer;
		struct raw_test_buffer reconstructed_buffer;
		struct spa_buffer *inputs[6] = { NULL };
		struct spa_buffer *outputs[3];

		init_raw_buffer(&raw_buffer, raw_values, sizeof(raw_values));
		for (uint32_t row = 0; row < 64; row++) {
			reconstructor_values[row * 12 + row % 12] = 1.0f;
			reconstructed_values[row] = -1.0f;
		}
		init_raw_buffer(&reconstructor_buffer, reconstructor_values,
				sizeof(reconstructor_values));
		init_raw_buffer(&reconstruction_buffer, reconstruction_values,
				sizeof(reconstruction_values));
		init_raw_buffer(&mean_buffer, mean_value, sizeof(mean_value));
		init_raw_buffer(&reconstructed_buffer, reconstructed_values,
				sizeof(reconstructed_values));
		inputs[0] = &raw_buffer.buffer;
		outputs[0] = &reconstruction_buffer.buffer;
		outputs[1] = &mean_buffer.buffer;
		outputs[2] = &reconstructed_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_update_parameter(graph, 5,
				&reconstructor_buffer.buffer) == 0);
		for (uint64_t sequence = 1; sequence <= 2; sequence++) {
			for (uint32_t first_row = 0; first_row < 8; first_row += 2) {
				for (uint32_t local = 0; local < 16; local++)
					raw_values[local] = 1u + first_row * 8u + local;
				raw_buffer.header.seq = sequence;
				raw_buffer.header.offset = first_row;
				raw_buffer.header.pts = first_row == 6 ? 246810 :
					SPA_TIME_INVALID;
				raw_buffer.header.flags = first_row == 6 ?
					SPA_META_HEADER_FLAG_MARKER : 0;
				assert(spa_fgn_graph_process(graph, inputs, 6,
						outputs, 3) == (sequence == 1 && first_row == 0 ?
							SPA_FGN_PROCESS_RESULT_PROPS_CHANGED : 0));
				if (first_row < 6) {
					assert(reconstruction_buffer.chunk.size == 0);
					assert(mean_buffer.chunk.size == 0);
					assert(reconstructed_buffer.chunk.size == 0);
				}
			}
			assert(reconstruction_buffer.chunk.size ==
					sizeof(reconstruction_values));
			assert(mean_buffer.chunk.size == sizeof(mean_value));
			assert(reconstructed_buffer.chunk.size ==
					sizeof(reconstructed_values));
			assert(fabsf(mean_value[0] - 31.5f) < 1.0e-6f);
			for (uint32_t pixel = 0; pixel < 12; pixel++) {
				float expected = sequence == 1 ? 0.0f :
					(float)selected[pixel] / 31.5f;
				assert(fabsf(reconstruction_values[pixel] - expected) <
						1.0e-6f);
			}
			for (uint32_t row = 0; row < 64; row++) {
				float expected = sequence == 1 ? 0.0f :
					(float)selected[row % 12] / 31.5f;
				assert(fabsf(reconstructed_values[row] - expected) <
						1.0e-6f);
			}
			assert(reconstruction_buffer.header.seq == sequence);
			assert(mean_buffer.header.seq == sequence);
			assert(reconstructed_buffer.header.seq == sequence);
			assert(reconstruction_buffer.header.pts == 246810);
			assert(mean_buffer.header.pts == 246810);
			assert(reconstructed_buffer.header.pts == 246810);
		}
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_detector_frame_assembler(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[4096];
	int res;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = assemble plugin = \"%s\""
		"   label = detector-frame-assembler-f32"
		" config = { detector_size = [ 4 4 ] "
		"              region_lines = 2 block_lines = 1 region_samples = 4"
		"              detector_origins = [ [ 0 0 ] [ 3 0 ] ]"
		"              detector_line_steps = [ [ 1 0 ] [ -1 0 ] ]"
		"              detector_sample_steps = [ [ 0 1 ] [ 0 1 ] ]"
		"              rate = [ 10 1 ] } }"
		"] inputs = [ \"assemble:readout-block\" ]"
		" outputs = [ \"assemble:detector-frame\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float block_values[8] = { 1, 2, 3, 4, 13, 14, 15, 16 };
		float frame_values[16];
		struct raw_test_buffer block_buffer, frame_buffer;
		struct spa_buffer *inputs[1], *outputs[1];

		for (uint32_t index = 0; index < 16; index++)
			frame_values[index] = -1.0f;
		init_raw_buffer(&block_buffer, block_values, sizeof(block_values));
		init_raw_buffer(&frame_buffer, frame_values, sizeof(frame_values));
		inputs[0] = &block_buffer.buffer;
		outputs[0] = &frame_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		block_buffer.header.seq = 81;
		block_buffer.header.offset = 0;
		block_buffer.header.flags = 0;
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(frame_buffer.chunk.size == 0);
		for (uint32_t index = 0; index < 16; index++)
			assert(frame_values[index] == -1.0f);
		memcpy(block_values, (float[]){ 5, 6, 7, 8, 9, 10, 11, 12 },
			sizeof(block_values));
		block_buffer.header.offset = 1;
		block_buffer.header.flags = SPA_META_HEADER_FLAG_MARKER;
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(frame_buffer.chunk.size == sizeof(frame_values));
		for (uint32_t index = 0; index < 16; index++)
			assert(frame_values[index] == (float)(index + 1));
		assert(frame_buffer.header.seq == 81 && frame_buffer.header.offset == 0);
		assert(frame_buffer.header.flags & SPA_META_HEADER_FLAG_MARKER);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_pixel_calibration_readout(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[4096];
	int res;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = calibrate plugin = \"%s\""
		"   label = pixel-calibration-readout-u16-f32"
		" config = { detector_size = [ 4 4 ] "
		"              region_lines = 2 block_lines = 1 region_samples = 4"
		"              detector_origins = [ [ 0 0 ] [ 3 0 ] ]"
		"              detector_line_steps = [ [ 1 0 ] [ -1 0 ] ]"
		"              detector_sample_steps = [ [ 0 1 ] [ 0 1 ] ]"
		"              rate = [ 10 1 ]  initial_flat = null  initial_background = null } }"
		" { type = ndarray name = assemble plugin = \"%s\""
		"   label = detector-frame-assembler-f32"
		" config = { detector_size = [ 4 4 ] "
		"              region_lines = 2 block_lines = 1 region_samples = 4"
		"              detector_origins = [ [ 0 0 ] [ 3 0 ] ]"
		"              detector_line_steps = [ [ 1 0 ] [ -1 0 ] ]"
		"              detector_sample_steps = [ [ 0 1 ] [ 0 1 ] ]"
		"              rate = [ 10 1 ] } }"
		"] links = [ { output = \"calibrate:calibrated\""
		"              input = \"assemble:readout-block\" } ]"
		" inputs = [ \"calibrate:raw\" \"calibrate:flat\""
		"            \"calibrate:background\" ]"
		" outputs = [ \"calibrate:calibrated\""
		"             \"assemble:detector-frame\" ] }", plugin, plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		uint16_t raw_values[8] = { 1, 2, 3, 4, 13, 14, 15, 16 };
		float calibrated_values[8];
		float frame_values[16];
		struct raw_test_buffer raw_buffer, calibrated_buffer, frame_buffer;
		struct spa_buffer *inputs[3], *outputs[2];

		for (uint32_t index = 0; index < 16; index++)
			frame_values[index] = -1.0f;
		init_raw_buffer(&raw_buffer, raw_values, sizeof(raw_values));
		init_raw_buffer(&calibrated_buffer, calibrated_values,
				sizeof(calibrated_values));
		init_raw_buffer(&frame_buffer, frame_values, sizeof(frame_values));
		inputs[0] = &raw_buffer.buffer;
		inputs[1] = NULL;
		inputs[2] = NULL;
		outputs[0] = &calibrated_buffer.buffer;
		outputs[1] = &frame_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		raw_buffer.header.seq = 82;
		raw_buffer.header.offset = 0;
		raw_buffer.header.flags = 0;
		assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 2) == 0);
		assert(calibrated_buffer.chunk.size == sizeof(calibrated_values));
		for (uint32_t index = 0; index < 8; index++)
			assert(calibrated_values[index] == (float)raw_values[index]);
		assert(calibrated_buffer.header.seq == 82 &&
				calibrated_buffer.header.offset == 0);
		assert(!(calibrated_buffer.header.flags & SPA_META_HEADER_FLAG_MARKER));
		assert(frame_buffer.chunk.size == 0);
		for (uint32_t index = 0; index < 16; index++)
			assert(frame_values[index] == -1.0f);

		memcpy(raw_values, (uint16_t[]){ 5, 6, 7, 8, 9, 10, 11, 12 },
			sizeof(raw_values));
		raw_buffer.header.offset = 1;
		raw_buffer.header.flags = SPA_META_HEADER_FLAG_MARKER;
		assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 2) == 0);
		assert(calibrated_buffer.chunk.size == sizeof(calibrated_values));
		for (uint32_t index = 0; index < 8; index++)
			assert(calibrated_values[index] == (float)raw_values[index]);
		assert(calibrated_buffer.header.seq == 82 &&
				calibrated_buffer.header.offset == 1);
		assert(calibrated_buffer.header.flags & SPA_META_HEADER_FLAG_MARKER);
		assert(frame_buffer.chunk.size == sizeof(frame_values));
		for (uint32_t index = 0; index < 16; index++)
			assert(frame_values[index] == (float)(index + 1));
		assert(frame_buffer.header.seq == 82 && frame_buffer.header.offset == 0);
		assert(frame_buffer.header.flags & SPA_META_HEADER_FLAG_MARKER);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_shwfs_readout_reconstructor(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[4096];
	int res;

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = reconstruct plugin = \"%s\""
		" label = shwfs-readout-reconstructor-f32"
		" config = { detector_size = [ 4 4 ] "
		" region_lines = 2 block_lines = 1 region_samples = 4"
		" detector_origins = [ [ 0 0 ] [ 3 0 ] ]"
		" detector_line_steps = [ [ 1 0 ] [ -1 0 ] ]"
		" detector_sample_steps = [ [ 0 1 ] [ 0 1 ] ]"
		" subaperture_size = [ 1 2 ] "
		" subaperture_count = 4 reconstructed_count = 1"
		" initial_subaperture_origins ="
		" [ [ 3 0 ] [ 0 0 ] [ 2 0 ] [ 1 0 ] ]"
		" coordinate_scale = 1.0 pixel_threshold = 0.0 flux_threshold = 0.0"
		" initial_reconstructor = [ 1 0 0 0 0 0 0 0 ]"
		" reconstructed_schema = test.shwfs-readout/1 rate = [ 10 1 ]  reference_slopes = null  active = null } } ]"
		" inputs = [ \"reconstruct:readout-block\" ]"
		" outputs = [ \"reconstruct:reconstructed\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float block_values[8] = { 1, 2, 3, 4, 13, 14, 15, 16 };
		float reconstructed_values[1] = { -1.0f };
		struct raw_test_buffer block_buffer, reconstructed_buffer;
		struct spa_buffer *inputs[1], *outputs[1];

		init_raw_buffer(&block_buffer, block_values, sizeof(block_values));
		init_raw_buffer(&reconstructed_buffer, reconstructed_values,
				sizeof(reconstructed_values));
		inputs[0] = &block_buffer.buffer;
		outputs[0] = &reconstructed_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		block_buffer.header.seq = 83;
		block_buffer.header.offset = 0;
		block_buffer.header.flags = 0;
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(reconstructed_buffer.chunk.size == 0);
		assert(reconstructed_values[0] == -1.0f);
		memcpy(block_values, (float[]){ 5, 6, 7, 8, 9, 10, 11, 12 },
			sizeof(block_values));
		block_buffer.header.offset = 1;
		block_buffer.header.flags = SPA_META_HEADER_FLAG_MARKER;
		assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
		assert(reconstructed_buffer.chunk.size == sizeof(reconstructed_values));
		assert(fabsf(reconstructed_values[0] - 1.0f / 54.0f) < 1.0e-6f);
		assert(reconstructed_buffer.header.seq == 83 &&
				reconstructed_buffer.header.offset == 0);
		assert(reconstructed_buffer.header.flags & SPA_META_HEADER_FLAG_MARKER);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_pwfs_readout_reconstructor(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[4096];
	int res;

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = reconstruct plugin = \"%s\""
		" label = pwfs-readout-reconstructor-f32"
		" config = { detector_size = [ 4 4 ] "
		" region_lines = 2 block_lines = 1 region_samples = 4"
		" detector_origins = [ [ 0 0 ] [ 3 0 ] ]"
		" detector_line_steps = [ [ 1 0 ] [ -1 0 ] ]"
		" detector_sample_steps = [ [ 0 1 ] [ 0 1 ] ]"
		" pupil_size = [ 1 1 ] "
		" pupil_origins = [ [ 3 0 ] [ 0 0 ] [ 2 0 ] [ 1 0 ] ]"
		" pupil_mask = [ true ] reconstructed_count = 1"
		" initial_reconstructor = [ 1 1 1 1 ]"
		" reconstructed_schema = test.pwfs-readout/1 rate = [ 10 1 ] } } ]"
		" inputs = [ \"reconstruct:readout-block\" ]"
		" outputs = [ \"reconstruct:reconstruction-pixels\""
		" \"reconstruct:mean-pupil-intensity\" \"reconstruct:reconstructed\" ] }",
		plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float block_values[8] = { 1, 2, 3, 4, 13, 14, 15, 16 };
		float pixels[4] = { -1.0f, -1.0f, -1.0f, -1.0f };
		float mean[1] = { -1.0f };
		float reconstructed[1] = { -1.0f };
		struct raw_test_buffer block_buffer, pixels_buffer, mean_buffer;
		struct raw_test_buffer reconstructed_buffer;
		struct spa_buffer *inputs[1], *outputs[3];

		init_raw_buffer(&block_buffer, block_values, sizeof(block_values));
		init_raw_buffer(&pixels_buffer, pixels, sizeof(pixels));
		init_raw_buffer(&mean_buffer, mean, sizeof(mean));
		init_raw_buffer(&reconstructed_buffer, reconstructed,
				sizeof(reconstructed));
		inputs[0] = &block_buffer.buffer;
		outputs[0] = &pixels_buffer.buffer;
		outputs[1] = &mean_buffer.buffer;
		outputs[2] = &reconstructed_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		for (uint64_t sequence = 84; sequence <= 85; sequence++) {
			block_buffer.header.seq = sequence;
			block_buffer.header.offset = 0;
			block_buffer.header.flags = 0;
			assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 3) == 0);
			assert(pixels_buffer.chunk.size == 0 && mean_buffer.chunk.size == 0 &&
					reconstructed_buffer.chunk.size == 0);
			memcpy(block_values,
				(float[]){ 5, 6, 7, 8, 9, 10, 11, 12 }, sizeof(block_values));
			block_buffer.header.offset = 1;
			block_buffer.header.flags = SPA_META_HEADER_FLAG_MARKER;
			assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 3) == 0);
			assert(pixels_buffer.chunk.size == sizeof(pixels));
			assert(mean_buffer.chunk.size == sizeof(mean));
			assert(reconstructed_buffer.chunk.size == sizeof(reconstructed));
			assert(mean[0] == 7.0f);
			if (sequence == 84) {
				for (uint32_t index = 0; index < 4; index++)
					assert(pixels[index] == 0.0f);
				assert(reconstructed[0] == 0.0f);
			} else {
				assert(fabsf(pixels[0] - 13.0f / 7.0f) < 1.0e-6f);
				assert(fabsf(pixels[1] - 1.0f / 7.0f) < 1.0e-6f);
				assert(fabsf(pixels[2] - 9.0f / 7.0f) < 1.0e-6f);
				assert(fabsf(pixels[3] - 5.0f / 7.0f) < 1.0e-6f);
				assert(fabsf(reconstructed[0] - 4.0f) < 1.0e-6f);
			}
			assert(reconstructed_buffer.header.seq == sequence &&
					reconstructed_buffer.header.offset == 0);
			assert(reconstructed_buffer.header.flags & SPA_META_HEADER_FLAG_MARKER);
			memcpy(block_values,
				(float[]){ 1, 2, 3, 4, 13, 14, 15, 16 }, sizeof(block_values));
		}
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

static void test_pdm_system_flat(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[2048];
	int res;

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = flat plugin = \"%s\""
		" label = pdm-system-flat-f32 config = { actuator_count = 2 } } ]"
		" inputs = [ \"flat:requested\" \"flat:system-flat\" ]"
		" outputs = [ \"flat:requested-with-system-flat\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	{
		float requested_values[2] = { 1.0f, 2.0f };
		float system_flat_values[2] = { 0.5f, -1.0f };
		float output_values[2] = { 0.0f };
		struct raw_test_buffer requested_buffer, system_flat_buffer, output_buffer;
		struct spa_buffer *inputs[2], *outputs[1];

		init_raw_buffer(&requested_buffer, requested_values, sizeof(requested_values));
		init_raw_buffer(&system_flat_buffer, system_flat_values, sizeof(system_flat_values));
		init_raw_buffer(&output_buffer, output_values, sizeof(output_values));
		inputs[0] = &requested_buffer.buffer;
		inputs[1] = NULL;
		outputs[0] = &output_buffer.buffer;
		requested_buffer.header.seq = 91;
		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) == 0);
		assert(output_values[0] == 1.0f && output_values[1] == 2.0f);
		assert(spa_fgn_graph_update_parameter(graph, 1, &system_flat_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 1) ==
			SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(output_values[0] == 1.5f && output_values[1] == 1.0f);
		assert(output_buffer.header.seq == 91);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
}

#include "catalog_additions_declarations.h"
#include "utility_declarations.h"
#include "scientific_contract_declarations.h"
#include "control_declarations.h"
#include "shape_dense_ml_declarations.h"
#include "shack_streaming_declarations.h"
#include "pupil_signal_declarations.h"
#include "scc_processing_declarations.h"
#include "pyramid_composite_declarations.h"
#include "dm_failure_declarations.h"
#include "streaming_metadata_declarations.h"

int main(int argc, char *argv[])
{
	struct spa_fgn_graph *graph = NULL;
	const struct spa_fgn_format *format;
	struct test_buffer input, output;
	struct spa_buffer *inputs[1], *outputs[1];
	char config[4096];
	int res;

	assert(argc == 2);
	test_catalog_additions_declarations(argv[1]);
	test_remaining_production_declarations(argv[1]);
	test_row_reconstructor(argv[1]);
	test_raw_row_front_end(argv[1]);
	test_raw_row_pwfs_normalization(argv[1]);
	test_raw_row_pwfs_front_end(argv[1]);
	test_detector_frame_assembler(argv[1]);
	test_pixel_calibration_readout(argv[1]);
	test_shwfs_readout_reconstructor(argv[1]);
	test_pwfs_readout_reconstructor(argv[1]);
	test_pdm_system_flat(argv[1]);
	test_ndarray_copy(argv[1]);
	test_scientific_contract_declarations(argv[1]);
	test_acquisition_chopped_declaration(argv[1]);
	test_vector_arithmetic(argv[1]);
	test_shape_dense_ml_declarations(argv[1]);
	test_shack_streaming_declarations(argv[1]);
	test_pupil_signal_declarations(argv[1]);
	test_scc_processing_declarations(argv[1]);
	test_pyramid_composite_declarations(argv[1]);
	test_dm_failure_declarations(argv[1]);
	test_streaming_metadata_declarations(argv[1]);
	test_command_conditioning_dac_configuration(argv[1]);
	test_conditioning_feedback_overflow(argv[1]);
	test_signed_pixel_and_biquad(argv[1]);
	test_closed_loop_integrator(argv[1]);
	test_weighted_centroid(argv[1]);
	test_corrector_allocation(argv[1]);
	test_shack_summary_and_weighted_image(argv[1]);
	test_scc_residual_and_reconstruction(argv[1]);
	test_frame_delay_and_vector_conditioning(argv[1]);
	test_pdm_engineering_offset(argv[1]);
	test_leaky_rejection_and_reset(argv[1]);
	test_pyramid_pupil_flux(argv[1]);
	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = integrator plugin = \"%s\""
		"   label = leaky-integrator-f32"
		" config = { extent = 2 initial_state = 4.0"
		"              input_schema = pipewireao/controller-residual-error-f32/1"
		"              output_schema = pipewireao/controller-command-f32/1 }"
		"   props = { gain = 0.5 pole = 0.75 } }"
		"] inputs = [ \"integrator:input\" ]"
		" outputs = [ \"integrator:output\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "declaration graph construction failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 1);
	assert(spa_fgn_graph_get_n_outputs(graph) == 1);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE);
	assert(format->n_dimensions == 1 && format->shape[0] == 2);
	assert(strcmp(format->schema,
			"pipewireao/controller-residual-error-f32/1") == 0);

	init_buffer(&input);
	init_buffer(&output);
	input.values[0] = 2.0f;
	input.values[1] = 2.0f;
	input.header.seq = 41;
	inputs[0] = &input.buffer;
	outputs[0] = &output.buffer;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) ==
			SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	assert(output.values[0] == 4.0f && output.values[1] == 4.0f);
	assert(output.header.seq == 41);

	input.values[0] = 4.0f;
	input.values[1] = -4.0f;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(output.values[0] == 5.0f && output.values[1] == 1.0f);

	outputs[0] = &input.buffer;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == -EINVAL);
	outputs[0] = &output.buffer;

	assert(spa_fgn_graph_reset(graph) == 0);
	input.values[0] = 0.0f;
	input.values[1] = 0.0f;
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(output.values[0] == 3.0f && output.values[1] == 3.0f);
	assert(spa_fgn_graph_deactivate(graph) == 0);
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = dither-chain plugin = \"%s\""
		"   label = optical-gain-dither-f32"
		"   config = { modes = [ { amplitude = 0.5 frames_per_period = 4"
		"                            periods_per_estimate = 1 } ]"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = estimator plugin = \"%s\""
		"   label = optical-gain-estimator-f32"
		"   config = { modes = [ { amplitude = 0.5 frames_per_period = 4"
		"                            periods_per_estimate = 1 } ]"
		"              initial_calibration_responses = [ 1.0 ]"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = gain-update plugin = \"%s\""
		"   label = optical-gain-update-f32"
		"   config = { criteria = [ { minimum_command_amplitude = 0.1"
		"                               minimum_response_amplitude = 0.05"
		"                               minimum_optical_gain = 0.1"
		"                               maximum_optical_gain = 2.0"
		"                               maximum_absolute_phase_lag = 0.25 } ]"
		"              assignments = [ { kind = \"dither-mode\" dither_mode = 0"
		"                                  update_fraction = 1.0 }"
		"                                { kind = hold } ]"
		"              initial_optical_gains = [ 1.0 3.0 ]"
		"              rate = [ 1000 1 ] } }"
		"] links = ["
		" { output = \"dither-chain:commanded-modes\""
		"   input = \"estimator:realized-command-modes\" }"
		" { output = \"dither-chain:window-frame\""
		"   input = \"estimator:window-frame\" }"
		" { output = \"dither-chain:phase-indices\""
		"   input = \"estimator:phase-indices\" }"
		" { output = \"dither-chain:completes-window\""
		"   input = \"estimator:completes-window\" }"
		" { output = \"estimator:command-amplitudes\""
		"   input = \"gain-update:command-amplitudes\" }"
		" { output = \"estimator:response-amplitudes\""
		"   input = \"gain-update:response-amplitudes\" }"
		" { output = \"estimator:optical-gains\""
		"   input = \"gain-update:optical-gain-estimates\" }"
		" { output = \"estimator:phase-lags\""
		"   input = \"gain-update:phase-lags\" }"
		" { output = \"estimator:valid\""
		"   input = \"gain-update:valid\" }"
		" { output = \"estimator:estimate-complete\""
		"   input = \"gain-update:estimate-complete\" }"
		"] inputs = [ \"estimator:reconstructed-response-modes\""
		"             \"estimator:calibration-responses\" ]"
		" outputs = [ \"gain-update:optical-gains\""
		"             \"gain-update:accepted-dither-modes\""
		"             \"gain-update:updated-reconstructed-modes\" ] }",
		argv[1], argv[1], argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "optical-gain estimator graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 3);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 1 && format->shape[0] == 2);
	assert(strcmp(format->schema,
			"pipewireao/physical-optical-gain-f32/1") == 0);

	{
		const float responses[4] = { 0.25f, 0.0f, -0.25f, 0.0f };
		float response_value[1] = { 0.0f };
		float replacement_value[1] = { 0.5f };
		float optical_gain_values[2] = { 0.0f };
		uint8_t accepted_dither_value[1] = { 1 };
		uint8_t updated_reconstructed_values[2] = { 1, 1 };
		struct raw_test_buffer response_buffer, replacement_buffer;
		struct raw_test_buffer optical_gain_buffer, accepted_dither_buffer;
		struct raw_test_buffer updated_reconstructed_buffer;
		struct spa_buffer *estimator_inputs[2];
		struct spa_buffer *update_outputs[3];

		init_raw_buffer(&response_buffer, response_value,
				sizeof(response_value));
		init_raw_buffer(&replacement_buffer, replacement_value,
				sizeof(replacement_value));
		init_raw_buffer(&optical_gain_buffer, optical_gain_values,
				sizeof(optical_gain_values));
		init_raw_buffer(&accepted_dither_buffer, accepted_dither_value,
				sizeof(accepted_dither_value));
		init_raw_buffer(&updated_reconstructed_buffer,
				updated_reconstructed_values,
				sizeof(updated_reconstructed_values));
		response_buffer.header.seq = 84;
		estimator_inputs[0] = &response_buffer.buffer;
		estimator_inputs[1] = NULL;
		update_outputs[0] = &optical_gain_buffer.buffer;
		update_outputs[1] = &accepted_dither_buffer.buffer;
		update_outputs[2] = &updated_reconstructed_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		for (size_t frame = 0; frame < 4; frame++) {
			response_value[0] = responses[frame];
			assert(spa_fgn_graph_process(graph, estimator_inputs, 2,
					update_outputs, 3) == 0);
			if (frame < 3) {
				assert(optical_gain_values[0] == 1.0f &&
						optical_gain_values[1] == 3.0f);
				assert(accepted_dither_value[0] == 0 &&
						updated_reconstructed_values[0] == 0 &&
						updated_reconstructed_values[1] == 0);
			}
		}
		assert(fabsf(optical_gain_values[0] - 0.5f) < 2.0e-6f &&
				optical_gain_values[1] == 3.0f);
		assert(accepted_dither_value[0] == 1 &&
				updated_reconstructed_values[0] == 1 &&
				updated_reconstructed_values[1] == 0);
		for (size_t output_index = 0; output_index < 3; output_index++) {
			struct spa_meta_header *header =
					update_outputs[output_index]->metas[0].data;
			assert(header->seq == 84);
		}

		assert(spa_fgn_graph_update_parameter(graph, 1,
				&replacement_buffer.buffer) == 0);
		for (size_t frame = 0; frame < 4; frame++) {
			int expected_status = frame == 0
					? SPA_FGN_PROCESS_RESULT_PROPS_CHANGED : 0;
			response_value[0] = responses[frame];
			assert(spa_fgn_graph_process(graph, estimator_inputs, 2,
					update_outputs, 3) == expected_status);
		}
		assert(fabsf(optical_gain_values[0] - 1.0f) < 2.0e-6f &&
				optical_gain_values[1] == 3.0f);
		assert(spa_fgn_graph_reset(graph) == 0);
		response_value[0] = responses[0];
		assert(spa_fgn_graph_process(graph, estimator_inputs, 2,
				update_outputs, 3) == 0);
		assert(optical_gain_values[0] == 1.0f &&
				optical_gain_values[1] == 3.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = dither-mismatch plugin = \"%s\""
		"   label = optical-gain-dither-f32"
		"   config = { modes = [ { amplitude = 0.5 frames_per_period = 4"
		"                            periods_per_estimate = 2 } ] } }"
		" { type = ndarray name = estimator-mismatch plugin = \"%s\""
		"   label = optical-gain-estimator-f32"
		"   config = { modes = [ { amplitude = 0.5 frames_per_period = 8"
		"                            periods_per_estimate = 1 } ]"
		"              initial_calibration_responses = [ 1.0 ] } }"
		"] links = ["
		" { output = \"dither-mismatch:commanded-modes\""
		"   input = \"estimator-mismatch:realized-command-modes\" }"
		" { output = \"dither-mismatch:window-frame\""
		"   input = \"estimator-mismatch:window-frame\" }"
		" { output = \"dither-mismatch:phase-indices\""
		"   input = \"estimator-mismatch:phase-indices\" }"
		" { output = \"dither-mismatch:completes-window\""
		"   input = \"estimator-mismatch:completes-window\" }"
		"] inputs = [ \"estimator-mismatch:reconstructed-response-modes\""
		"             \"estimator-mismatch:calibration-responses\" ]"
		" outputs = [ \"estimator-mismatch:estimate-complete\" ] }",
		argv[1], argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "optical-gain schedule-mismatch graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	{
		float response_value[1] = { 0.0f };
		uint8_t estimate_complete_value[1] = { 1 };
		struct raw_test_buffer response_buffer, estimate_complete_buffer;
		struct spa_buffer *mismatch_inputs[2];
		struct spa_buffer *mismatch_outputs[1];

		init_raw_buffer(&response_buffer, response_value,
				sizeof(response_value));
		init_raw_buffer(&estimate_complete_buffer, estimate_complete_value,
				sizeof(estimate_complete_value));
		mismatch_inputs[0] = &response_buffer.buffer;
		mismatch_inputs[1] = NULL;
		mismatch_outputs[0] = &estimate_complete_buffer.buffer;
		assert(spa_fgn_graph_activate(graph) == 0);
		for (size_t frame = 0; frame < 4; frame++)
			assert(spa_fgn_graph_process(graph, mismatch_inputs, 2,
					mismatch_outputs, 1) == 0);
		assert(spa_fgn_graph_process(graph, mismatch_inputs, 2,
				mismatch_outputs, 1) == -EINVAL);
		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, mismatch_inputs, 2,
				mismatch_outputs, 1) == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = optical-gain-dither plugin = \"%s\""
		"   label = optical-gain-dither-f32"
		"   config = { modes = [ { amplitude = 0.5 frames_per_period = 4"
		"                            periods_per_estimate = 1 } ]"
		"              rate = [ 1000 1 ] } }"
		"] outputs = [ \"optical-gain-dither:commanded-modes\""
		"             \"optical-gain-dither:window-frame\""
		"             \"optical-gain-dither:phase-indices\""
		"             \"optical-gain-dither:completes-window\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "optical-gain dither graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 0);
	assert(spa_fgn_graph_get_n_outputs(graph) == 4);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 1 && format->shape[0] == 1);
	assert(strcmp(format->schema,
			"pipewireao/optical-gain-dither-command-f32/1") == 0);

	{
		float command_value[1] = { 0.0f };
		uint64_t window_frame_value[1] = { UINT64_MAX };
		uint64_t phase_index_value[1] = { UINT64_MAX };
		uint8_t completes_window_value[1] = { UINT8_MAX };
		struct raw_test_buffer command_buffer, window_frame_buffer;
		struct raw_test_buffer phase_index_buffer;
		struct raw_test_buffer completes_window_buffer;
		struct spa_buffer *dither_outputs[4];
		const float expected_commands[4] = { 0.5f, 0.0f, -0.5f, 0.0f };

		init_raw_buffer(&command_buffer, command_value, sizeof(command_value));
		init_raw_buffer(&window_frame_buffer, window_frame_value,
				sizeof(window_frame_value));
		init_raw_buffer(&phase_index_buffer, phase_index_value,
				sizeof(phase_index_value));
		init_raw_buffer(&completes_window_buffer, completes_window_value,
				sizeof(completes_window_value));
		dither_outputs[0] = &command_buffer.buffer;
		dither_outputs[1] = &window_frame_buffer.buffer;
		dither_outputs[2] = &phase_index_buffer.buffer;
		dither_outputs[3] = &completes_window_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		for (uint64_t frame = 0; frame < 4; frame++) {
			assert(spa_fgn_graph_process(graph, NULL, 0,
					dither_outputs, 4) == 0);
			assert(fabsf(command_value[0] - expected_commands[frame]) <
					2.0f * FLT_EPSILON);
			assert(window_frame_value[0] == frame);
			assert(phase_index_value[0] == frame);
			assert(completes_window_value[0] == (frame == 3));
		}
		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, NULL, 0, dither_outputs, 4) == 0);
		assert(command_value[0] == 0.5f && window_frame_value[0] == 0 &&
				phase_index_value[0] == 0 && completes_window_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = stability plugin = \"%s\""
		"   label = interaction-matrix-stability-f32"
		" config = { measurement_count = 2 corrector_coordinate_count = 2"
		"              minimum_column_norm = 0.1"
		"              interaction_matrix_schema = test.filtergraph/test-interaction-matrix/1"
		"              rate = [ 1000 1 ] }"
		"   props = { \"maximum-relative-column-difference\" = 1.0"
		"             \"maximum-relative-frobenius-difference\" = 0.5 } }"
		"] inputs = [ \"stability:first-window\""
		"             \"stability:second-window\" ]"
		" outputs = [ \"stability:status\""
		"             \"stability:status-coordinate\""
		"             \"stability:minimum-column-norms\""
		"             \"stability:maximum-relative-column-difference\""
		"             \"stability:maximum-relative-column-difference-coordinate\""
		"             \"stability:relative-frobenius-difference\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "interaction-matrix stability graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 6);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_U8 &&
			format->n_dimensions == 1 && format->shape[0] == 1);
	assert(strcmp(format->schema,
			"pipewireao/interaction-matrix-stability-status-u8/1") == 0);

	{
		float first_window_values[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
		float second_window_values[4] = { 2.0f, 0.0f, 0.0f, 2.0f };
		uint8_t status_value[1] = { UINT8_MAX };
		uint64_t status_coordinate_value[1] = { UINT64_MAX };
		float minimum_column_norm_values[2] = { 0.0f };
		float maximum_column_difference_value[1] = { 0.0f };
		uint64_t maximum_column_coordinate_value[1] = { UINT64_MAX };
		float relative_frobenius_difference_value[1] = { 0.0f };
		struct raw_test_buffer first_window_buffer, second_window_buffer;
		struct raw_test_buffer status_buffer, status_coordinate_buffer;
		struct raw_test_buffer minimum_column_norm_buffer;
		struct raw_test_buffer maximum_column_difference_buffer;
		struct raw_test_buffer maximum_column_coordinate_buffer;
		struct raw_test_buffer relative_frobenius_difference_buffer;
		struct spa_buffer *stability_inputs[2];
		struct spa_buffer *stability_outputs[6];

		init_raw_buffer(&first_window_buffer, first_window_values,
				sizeof(first_window_values));
		init_raw_buffer(&second_window_buffer, second_window_values,
				sizeof(second_window_values));
		init_raw_buffer(&status_buffer, status_value, sizeof(status_value));
		init_raw_buffer(&status_coordinate_buffer, status_coordinate_value,
				sizeof(status_coordinate_value));
		init_raw_buffer(&minimum_column_norm_buffer, minimum_column_norm_values,
				sizeof(minimum_column_norm_values));
		init_raw_buffer(&maximum_column_difference_buffer,
				maximum_column_difference_value,
				sizeof(maximum_column_difference_value));
		init_raw_buffer(&maximum_column_coordinate_buffer,
				maximum_column_coordinate_value,
				sizeof(maximum_column_coordinate_value));
		init_raw_buffer(&relative_frobenius_difference_buffer,
				relative_frobenius_difference_value,
				sizeof(relative_frobenius_difference_value));
		first_window_buffer.header.seq = 60;
		second_window_buffer.header.seq = 61;
		stability_inputs[0] = &first_window_buffer.buffer;
		stability_inputs[1] = &second_window_buffer.buffer;
		stability_outputs[0] = &status_buffer.buffer;
		stability_outputs[1] = &status_coordinate_buffer.buffer;
		stability_outputs[2] = &minimum_column_norm_buffer.buffer;
		stability_outputs[3] = &maximum_column_difference_buffer.buffer;
		stability_outputs[4] = &maximum_column_coordinate_buffer.buffer;
		stability_outputs[5] = &relative_frobenius_difference_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, stability_inputs, 2,
				stability_outputs, 6) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(status_value[0] == 3 && status_coordinate_value[0] == 0);
		assert(minimum_column_norm_values[0] == 1.0f &&
				minimum_column_norm_values[1] == 2.0f);
		assert(fabsf(maximum_column_difference_value[0] - 2.0f / 3.0f) <
				1.0e-6f);
		assert(maximum_column_coordinate_value[0] == 0);
		assert(fabsf(relative_frobenius_difference_value[0] - 2.0f / 3.0f) <
				1.0e-6f);
		for (size_t i = 0; i < 6; i++) {
			struct spa_meta_header *header = stability_outputs[i]->metas[0].data;
			assert(header->seq == 61);
		}
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = difference plugin = \"%s\""
		"   label = matched-calibration-differencing-f32"
		" config = { measurement_count = 2 corrector_coordinate_count = 1"
		"              measurements_schema = test.filtergraph/test-measurements/1"
		"              corrector_sample_schema = test.filtergraph/test-corrector-sample/1"
		"              filtered_measurements_schema = test.filtergraph/test-filtered-measurements/1"
		"              filtered_corrector_sample_schema = test.filtergraph/test-filtered-corrector-sample/1"
		"              rate = [ 1000 1 ] } }"
		"] inputs = [ \"difference:measurements\""
		"             \"difference:corrector-sample\" ]"
		" outputs = [ \"difference:filtered-measurements\""
		"             \"difference:filtered-corrector-sample\""
		"             \"difference:filtered\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "matched calibration differencing graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 3);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			2, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_BOOL8 &&
			format->n_dimensions == 1 && format->shape[0] == 1);

	{
		float measurement_values[2] = { 10.0f, 20.0f };
		float corrector_values[1] = { 1.0f };
		float filtered_measurement_values[2] = { 9.0f, 9.0f };
		float filtered_corrector_values[1] = { 8.0f };
		uint8_t filtered_value[1] = { 1 };
		struct raw_test_buffer measurement_buffer, corrector_buffer;
		struct raw_test_buffer filtered_measurement_buffer;
		struct raw_test_buffer filtered_corrector_buffer, filtered_buffer;
		struct spa_buffer *difference_inputs[2];
		struct spa_buffer *difference_outputs[3];

		init_raw_buffer(&measurement_buffer, measurement_values,
				sizeof(measurement_values));
		init_raw_buffer(&corrector_buffer, corrector_values,
				sizeof(corrector_values));
		init_raw_buffer(&filtered_measurement_buffer,
				filtered_measurement_values,
				sizeof(filtered_measurement_values));
		init_raw_buffer(&filtered_corrector_buffer, filtered_corrector_values,
				sizeof(filtered_corrector_values));
		init_raw_buffer(&filtered_buffer, filtered_value, sizeof(filtered_value));
		measurement_buffer.header.seq = 72;
		difference_inputs[0] = &measurement_buffer.buffer;
		difference_inputs[1] = &corrector_buffer.buffer;
		difference_outputs[0] = &filtered_measurement_buffer.buffer;
		difference_outputs[1] = &filtered_corrector_buffer.buffer;
		difference_outputs[2] = &filtered_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, difference_inputs, 2,
				difference_outputs, 3) == 0);
		assert(filtered_measurement_values[0] == 0.0f &&
				filtered_measurement_values[1] == 0.0f);
		assert(filtered_corrector_values[0] == 0.0f && filtered_value[0] == 0);

		measurement_values[0] = 13.0f;
		measurement_values[1] = 14.0f;
		corrector_values[0] = 0.5f;
		assert(spa_fgn_graph_process(graph, difference_inputs, 2,
				difference_outputs, 3) == 0);
		assert(filtered_measurement_values[0] == 3.0f &&
				filtered_measurement_values[1] == -6.0f);
		assert(filtered_corrector_values[0] == -0.5f && filtered_value[0] == 1);
		assert(filtered_measurement_buffer.header.seq == 72 &&
				filtered_corrector_buffer.header.seq == 72 &&
				filtered_buffer.header.seq == 72);

		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, difference_inputs, 2,
				difference_outputs, 3) == 0);
		assert(filtered_measurement_values[0] == 0.0f &&
				filtered_measurement_values[1] == 0.0f);
		assert(filtered_corrector_values[0] == 0.0f && filtered_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = modal plugin = \"%s\""
		"   label = modal-push-pull-f32"
		" config = { mode_count = 2 corrector_coordinate_count = 2"
		"              initial_positive_commands = [ 1.0 0.0 0.0 2.0 ]"
		"              excitation_schema = test.filtergraph/test-modal-excitation/1"
		"              rate = [ 1000 1 ] } }"
		"] inputs = [ \"modal:positive-commands\" ]"
		" outputs = [ \"modal:excitation\""
		"             \"modal:pattern-index\""
		"             \"modal:polarity\""
		"             \"modal:completes-cycle\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "modal push-pull graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 1);
	assert(spa_fgn_graph_get_n_outputs(graph) == 4);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 2 && format->shape[0] == 2 &&
			format->shape[1] == 2);
	assert(strcmp(format->schema,
			"pipewireao/modal-positive-commands-f32/1") == 0);

	{
		float replacement_values[4] = { 3.0f, 0.0f, 0.0f, 4.0f };
		float excitation_values[2] = { 0.0f };
		uint64_t pattern_index_value[1] = { UINT64_MAX };
		uint8_t polarity_value[1] = { UINT8_MAX };
		uint8_t completes_cycle_value[1] = { 0 };
		struct raw_test_buffer replacement_buffer, excitation_buffer;
		struct raw_test_buffer pattern_index_buffer, polarity_buffer;
		struct raw_test_buffer completes_cycle_buffer;
		struct spa_buffer *modal_inputs[1] = { NULL };
		struct spa_buffer *modal_outputs[4];

		init_raw_buffer(&replacement_buffer, replacement_values,
				sizeof(replacement_values));
		init_raw_buffer(&excitation_buffer, excitation_values,
				sizeof(excitation_values));
		init_raw_buffer(&pattern_index_buffer, pattern_index_value,
				sizeof(pattern_index_value));
		init_raw_buffer(&polarity_buffer, polarity_value,
				sizeof(polarity_value));
		init_raw_buffer(&completes_cycle_buffer, completes_cycle_value,
				sizeof(completes_cycle_value));
		modal_outputs[0] = &excitation_buffer.buffer;
		modal_outputs[1] = &pattern_index_buffer.buffer;
		modal_outputs[2] = &polarity_buffer.buffer;
		modal_outputs[3] = &completes_cycle_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, modal_inputs, 1,
				modal_outputs, 4) == 0);
		assert(excitation_values[0] == 1.0f &&
				excitation_values[1] == 0.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 0 &&
				completes_cycle_value[0] == 0);

		assert(spa_fgn_graph_update_parameter(graph, 0,
				&replacement_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, modal_inputs, 1,
				modal_outputs, 4) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(excitation_values[0] == -3.0f &&
				excitation_values[1] == 0.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 1 &&
				completes_cycle_value[0] == 0);

		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, modal_inputs, 1,
				modal_outputs, 4) == 0);
		assert(excitation_values[0] == 3.0f &&
				excitation_values[1] == 0.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 0 &&
				completes_cycle_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = zonal plugin = \"%s\""
		"   label = zonal-push-pull-f32"
		" config = { amplitudes = [ 1.0 2.0 ]"
		"              excitation_schema = test.filtergraph/test-zonal-excitation/1"
		"              rate = [ 1000 1 ] } }"
		"] inputs = [ \"zonal:amplitudes\" ]"
		" outputs = [ \"zonal:excitation\""
		"             \"zonal:pattern-index\""
		"             \"zonal:polarity\""
		"             \"zonal:completes-cycle\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "zonal push-pull graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 1);
	assert(spa_fgn_graph_get_n_outputs(graph) == 4);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 1 && format->shape[0] == 2);
	assert(strcmp(format->schema,
			"pipewireao/calibration-probe-amplitudes-f32/1") == 0);

	{
		float replacement_values[2] = { 3.0f, 4.0f };
		float excitation_values[2] = { 0.0f };
		uint64_t pattern_index_value[1] = { UINT64_MAX };
		uint8_t polarity_value[1] = { UINT8_MAX };
		uint8_t completes_cycle_value[1] = { 0 };
		struct raw_test_buffer replacement_buffer, excitation_buffer;
		struct raw_test_buffer pattern_index_buffer, polarity_buffer;
		struct raw_test_buffer completes_cycle_buffer;
		struct spa_buffer *zonal_inputs[1] = { NULL };
		struct spa_buffer *zonal_outputs[4];

		init_raw_buffer(&replacement_buffer, replacement_values,
				sizeof(replacement_values));
		init_raw_buffer(&excitation_buffer, excitation_values,
				sizeof(excitation_values));
		init_raw_buffer(&pattern_index_buffer, pattern_index_value,
				sizeof(pattern_index_value));
		init_raw_buffer(&polarity_buffer, polarity_value,
				sizeof(polarity_value));
		init_raw_buffer(&completes_cycle_buffer, completes_cycle_value,
				sizeof(completes_cycle_value));
		zonal_outputs[0] = &excitation_buffer.buffer;
		zonal_outputs[1] = &pattern_index_buffer.buffer;
		zonal_outputs[2] = &polarity_buffer.buffer;
		zonal_outputs[3] = &completes_cycle_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, zonal_inputs, 1,
				zonal_outputs, 4) == 0);
		assert(excitation_values[0] == 1.0f &&
				excitation_values[1] == 0.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 0 &&
				completes_cycle_value[0] == 0);

		assert(spa_fgn_graph_update_parameter(graph, 0,
				&replacement_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, zonal_inputs, 1,
				zonal_outputs, 4) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(excitation_values[0] == -3.0f &&
				excitation_values[1] == 0.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 1 &&
				completes_cycle_value[0] == 0);

		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, zonal_inputs, 1,
				zonal_outputs, 4) == 0);
		assert(excitation_values[0] == 3.0f &&
				excitation_values[1] == 0.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 0 &&
				completes_cycle_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = hadamard plugin = \"%s\""
		"   label = hadamard-push-pull-f32"
		" config = { amplitudes = [ 1.0 2.0 3.0 ]"
		"              excitation_schema = test.filtergraph/test-hadamard-excitation/1"
		"              rate = [ 1000 1 ] } }"
		"] inputs = [ \"hadamard:amplitudes\" ]"
		" outputs = [ \"hadamard:excitation\""
		"             \"hadamard:pattern-index\""
		"             \"hadamard:polarity\""
		"             \"hadamard:completes-cycle\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "Hadamard push-pull graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 1);
	assert(spa_fgn_graph_get_n_outputs(graph) == 4);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 1 && format->shape[0] == 3);
	assert(strcmp(format->schema,
			"pipewireao/calibration-probe-amplitudes-f32/1") == 0);

	{
		float replacement_values[3] = { 3.0f, 4.0f, 5.0f };
		float excitation_values[3] = { 0.0f };
		uint64_t pattern_index_value[1] = { UINT64_MAX };
		uint8_t polarity_value[1] = { UINT8_MAX };
		uint8_t completes_cycle_value[1] = { 0 };
		struct raw_test_buffer replacement_buffer, excitation_buffer;
		struct raw_test_buffer pattern_index_buffer, polarity_buffer;
		struct raw_test_buffer completes_cycle_buffer;
		struct spa_buffer *hadamard_inputs[1] = { NULL };
		struct spa_buffer *hadamard_outputs[4];

		init_raw_buffer(&replacement_buffer, replacement_values,
				sizeof(replacement_values));
		init_raw_buffer(&excitation_buffer, excitation_values,
				sizeof(excitation_values));
		init_raw_buffer(&pattern_index_buffer, pattern_index_value,
				sizeof(pattern_index_value));
		init_raw_buffer(&polarity_buffer, polarity_value,
				sizeof(polarity_value));
		init_raw_buffer(&completes_cycle_buffer, completes_cycle_value,
				sizeof(completes_cycle_value));
		hadamard_outputs[0] = &excitation_buffer.buffer;
		hadamard_outputs[1] = &pattern_index_buffer.buffer;
		hadamard_outputs[2] = &polarity_buffer.buffer;
		hadamard_outputs[3] = &completes_cycle_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, hadamard_inputs, 1,
				hadamard_outputs, 4) == 0);
		assert(excitation_values[0] == 1.0f &&
				excitation_values[1] == 2.0f &&
				excitation_values[2] == 3.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 0 &&
				completes_cycle_value[0] == 0);

		assert(spa_fgn_graph_update_parameter(graph, 0,
				&replacement_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, hadamard_inputs, 1,
				hadamard_outputs, 4) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(excitation_values[0] == -3.0f &&
				excitation_values[1] == -4.0f &&
				excitation_values[2] == -5.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 1 &&
				completes_cycle_value[0] == 0);

		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, hadamard_inputs, 1,
				hadamard_outputs, 4) == 0);
		assert(excitation_values[0] == 3.0f &&
				excitation_values[1] == 4.0f &&
				excitation_values[2] == 5.0f);
		assert(pattern_index_value[0] == 0 && polarity_value[0] == 0 &&
				completes_cycle_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = zonal-decode plugin = \"%s\""
		"   label = zonal-interaction-matrix-f32"
		" config = { measurement_count = 2 amplitudes = [ 0.5 2.0 ]"
		"              response_schema = test.filtergraph/test-zonal-response-cycle/1"
		"              interaction_matrix_schema = test.filtergraph/test-interaction-matrix/1"
		"              rate = [ 1 1 ] } }"
		"] inputs = [ \"zonal-decode:responses\""
		"             \"zonal-decode:amplitudes\" ]"
		" outputs = [ \"zonal-decode:interaction-matrix\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "zonal interaction-matrix graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 1);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 2 && format->shape[0] == 4 &&
			format->shape[1] == 2);
	assert(strcmp(format->schema,
			"test.filtergraph/test-zonal-response-cycle/1") == 0);

	{
		float response_values[8] = {
			3.0f, 5.0f, 1.0f, 1.0f, 9.0f, -1.0f, 1.0f, 3.0f
		};
		float amplitude_values[2] = { 1.0f, 4.0f };
		float matrix_values[4] = { 0.0f };
		struct raw_test_buffer response_buffer, amplitude_buffer, matrix_buffer;
		struct spa_buffer *zonal_decode_inputs[2];
		struct spa_buffer *zonal_decode_outputs[1];

		init_raw_buffer(&response_buffer, response_values,
				sizeof(response_values));
		init_raw_buffer(&amplitude_buffer, amplitude_values,
				sizeof(amplitude_values));
		init_raw_buffer(&matrix_buffer, matrix_values, sizeof(matrix_values));
		response_buffer.header.seq = 88;
		zonal_decode_inputs[0] = &response_buffer.buffer;
		zonal_decode_inputs[1] = NULL;
		zonal_decode_outputs[0] = &matrix_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, zonal_decode_inputs, 2,
				zonal_decode_outputs, 1) == 0);
		assert(matrix_values[0] == 2.0f && matrix_values[1] == 2.0f &&
				matrix_values[2] == 4.0f && matrix_values[3] == -1.0f);
		assert(matrix_buffer.header.seq == 88);

		assert(spa_fgn_graph_update_parameter(graph, 1,
				&amplitude_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, zonal_decode_inputs, 2,
				zonal_decode_outputs, 1) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(matrix_values[0] == 1.0f && matrix_values[1] == 1.0f &&
				matrix_values[2] == 2.0f && matrix_values[3] == -0.5f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = hadamard-decode plugin = \"%s\""
		"   label = hadamard-interaction-matrix-f32"
		" config = { measurement_count = 1 amplitudes = [ 0.5 2.0 ]"
		"              response_schema = test.filtergraph/test-hadamard-response-cycle/1"
		"              interaction_matrix_schema = test.filtergraph/test-interaction-matrix/1"
		"              rate = [ 1 1 ] } }"
		"] inputs = [ \"hadamard-decode:responses\""
		"             \"hadamard-decode:amplitudes\" ]"
		" outputs = [ \"hadamard-decode:interaction-matrix\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "Hadamard interaction-matrix graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 1);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 2 && format->shape[0] == 8 &&
			format->shape[1] == 1);
	assert(strcmp(format->schema,
			"test.filtergraph/test-hadamard-response-cycle/1") == 0);

	{
		float response_values[8] = {
			6.0f, -6.0f, -2.0f, 2.0f, 4.0f, -4.0f, 0.0f, 0.0f
		};
		float amplitude_values[2] = { 1.0f, 4.0f };
		float matrix_values[2] = { 0.0f };
		struct raw_test_buffer response_buffer, amplitude_buffer, matrix_buffer;
		struct spa_buffer *hadamard_decode_inputs[2];
		struct spa_buffer *hadamard_decode_outputs[1];

		init_raw_buffer(&response_buffer, response_values,
				sizeof(response_values));
		init_raw_buffer(&amplitude_buffer, amplitude_values,
				sizeof(amplitude_values));
		init_raw_buffer(&matrix_buffer, matrix_values, sizeof(matrix_values));
		response_buffer.header.seq = 89;
		hadamard_decode_inputs[0] = &response_buffer.buffer;
		hadamard_decode_inputs[1] = NULL;
		hadamard_decode_outputs[0] = &matrix_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, hadamard_decode_inputs, 2,
				hadamard_decode_outputs, 1) == 0);
		assert(matrix_values[0] == 6.0f && matrix_values[1] == 0.0f);
		assert(matrix_buffer.header.seq == 89);

		assert(spa_fgn_graph_update_parameter(graph, 1,
				&amplitude_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, hadamard_decode_inputs, 2,
				hadamard_decode_outputs, 1) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(matrix_values[0] == 3.0f && matrix_values[1] == 0.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = covariance plugin = \"%s\""
		"   label = docrime-covariance-f32"
		" config = { measurement_count = 2 corrector_coordinate_count = 2"
		"              sample_limit = 2"
		"              measurements_schema = test.filtergraph/test-measurement/1"
		"              corrector_sample_schema = test.filtergraph/test-corrector-sample/1"
		"              cross_covariance_schema = test.filtergraph/test-cross-covariance/1"
		"              corrector_covariance_schema = test.filtergraph/test-corrector-covariance/1"
		"              rate = [ 1000 1 ] } }"
		"] inputs = [ \"covariance:measurements\""
		"             \"covariance:corrector-sample\" ]"
		" outputs = [ \"covariance:measurement-corrector-cross-covariance-sum\""
		"             \"covariance:corrector-covariance-sum\""
		"             \"covariance:sample-count\""
		"             \"covariance:complete\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "DO-CRIME covariance graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 4);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 2 && format->shape[0] == 2 &&
			format->shape[1] == 2);

	{
		float measurement_values[2] = { 10.0f, 4.5f };
		float corrector_values[2] = { 1.0f, 2.0f };
		float cross_covariance_values[4] = { 0.0f };
		float corrector_covariance_values[4] = { 0.0f };
		uint64_t sample_count_value[1] = { UINT64_MAX };
		uint8_t complete_value[1] = { 1 };
		struct raw_test_buffer measurement_buffer, corrector_buffer;
		struct raw_test_buffer cross_covariance_buffer;
		struct raw_test_buffer corrector_covariance_buffer;
		struct raw_test_buffer sample_count_buffer, complete_buffer;
		struct spa_buffer *covariance_inputs[2];
		struct spa_buffer *covariance_outputs[4];

		init_raw_buffer(&measurement_buffer, measurement_values,
				sizeof(measurement_values));
		init_raw_buffer(&corrector_buffer, corrector_values,
				sizeof(corrector_values));
		init_raw_buffer(&cross_covariance_buffer, cross_covariance_values,
				sizeof(cross_covariance_values));
		init_raw_buffer(&corrector_covariance_buffer,
				corrector_covariance_values,
				sizeof(corrector_covariance_values));
		init_raw_buffer(&sample_count_buffer, sample_count_value,
				sizeof(sample_count_value));
		init_raw_buffer(&complete_buffer, complete_value,
				sizeof(complete_value));
		measurement_buffer.header.seq = 90;
		covariance_inputs[0] = &measurement_buffer.buffer;
		covariance_inputs[1] = &corrector_buffer.buffer;
		covariance_outputs[0] = &cross_covariance_buffer.buffer;
		covariance_outputs[1] = &corrector_covariance_buffer.buffer;
		covariance_outputs[2] = &sample_count_buffer.buffer;
		covariance_outputs[3] = &complete_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, covariance_inputs, 2,
				covariance_outputs, 4) == 0);
		for (size_t i = 0; i < 4; i++) {
			assert(cross_covariance_values[i] == 0.0f);
			assert(corrector_covariance_values[i] == 0.0f);
		}
		assert(sample_count_value[0] == 1 && complete_value[0] == 0);

		measurement_values[0] = 16.0f;
		measurement_values[1] = -0.5f;
		corrector_values[0] = 3.0f;
		corrector_values[1] = 0.0f;
		assert(spa_fgn_graph_process(graph, covariance_inputs, 2,
				covariance_outputs, 4) == 0);
		assert(cross_covariance_values[0] == 6.0f &&
				cross_covariance_values[1] == -6.0f &&
				cross_covariance_values[2] == -5.0f &&
				cross_covariance_values[3] == 5.0f);
		assert(corrector_covariance_values[0] == 2.0f &&
				corrector_covariance_values[1] == -2.0f &&
				corrector_covariance_values[2] == -2.0f &&
				corrector_covariance_values[3] == 2.0f);
		assert(sample_count_value[0] == 2 && complete_value[0] == 1);
		assert(cross_covariance_buffer.header.seq == 90 &&
				corrector_covariance_buffer.header.seq == 90 &&
				sample_count_buffer.header.seq == 90 &&
				complete_buffer.header.seq == 90);

		assert(spa_fgn_graph_reset(graph) == 0);
		measurement_values[0] = 10.0f;
		measurement_values[1] = 4.5f;
		corrector_values[0] = 1.0f;
		corrector_values[1] = 2.0f;
		assert(spa_fgn_graph_process(graph, covariance_inputs, 2,
				covariance_outputs, 4) == 0);
		for (size_t i = 0; i < 4; i++) {
			assert(cross_covariance_values[i] == 0.0f);
			assert(corrector_covariance_values[i] == 0.0f);
		}
		assert(sample_count_value[0] == 1 && complete_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = open-loop-solve plugin = \"%s\""
		"   label = docrime-open-loop-interaction-matrix-f32"
		" config = { measurement_count = 2 corrector_coordinate_count = 2"
		"              minimum_corrector_variance = 0.1"
		"              cross_covariance_schema = test.filtergraph/test-cross-covariance/1"
		"              corrector_covariance_schema = test.filtergraph/test-corrector-covariance/1"
		"              interaction_matrix_schema = test.filtergraph/test-interaction-matrix/1"
		"              rate = [ 1 1 ] }"
		"   props = { \"maximum-absolute-corrector-correlation\" = 0.8"
		"             \"minimum-factorization-pivot-ratio\" = 0.02 } }"
		"] inputs = [ \"open-loop-solve:measurement-corrector-cross-covariance-sum\""
		"             \"open-loop-solve:corrector-covariance-sum\""
		"             \"open-loop-solve:sample-count\" ]"
		" outputs = [ \"open-loop-solve:interaction-matrix\""
		"             \"open-loop-solve:status\""
		"             \"open-loop-solve:status-coordinates\""
		"             \"open-loop-solve:diagnostic-sample-count\""
		"             \"open-loop-solve:corrector-variance-range\""
		"             \"open-loop-solve:maximum-absolute-corrector-correlation\""
		"             \"open-loop-solve:minimum-factorization-pivot-ratio\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "DO-CRIME open-loop solve graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 3);
	assert(spa_fgn_graph_get_n_outputs(graph) == 7);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			1, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_U8 &&
			format->n_dimensions == 1 && format->shape[0] == 1);
	assert(strcmp(format->schema,
			"pipewireao/docrime-interaction-matrix-status-u8/1") == 0);

	{
		float cross_covariance_values[4] = { 6.0f, 0.0f, 8.0f, 13.0f };
		float corrector_covariance_values[4] = { 4.0f, 2.0f, 2.0f, 4.0f };
		uint64_t sample_count_value[1] = { 3 };
		float matrix_values[4] = { 0.0f };
		uint8_t status_value[1] = { UINT8_MAX };
		uint64_t status_coordinate_values[2] = { UINT64_MAX, UINT64_MAX };
		uint64_t diagnostic_sample_count_value[1] = { UINT64_MAX };
		float variance_range_values[2] = { 0.0f };
		float correlation_value[1] = { 0.0f };
		float pivot_ratio_value[1] = { 0.0f };
		struct raw_test_buffer cross_covariance_buffer;
		struct raw_test_buffer corrector_covariance_buffer, sample_count_buffer;
		struct raw_test_buffer matrix_buffer, status_buffer;
		struct raw_test_buffer status_coordinate_buffer;
		struct raw_test_buffer diagnostic_sample_count_buffer;
		struct raw_test_buffer variance_range_buffer, correlation_buffer;
		struct raw_test_buffer pivot_ratio_buffer;
		struct spa_buffer *open_loop_inputs[3];
		struct spa_buffer *open_loop_outputs[7];

		init_raw_buffer(&cross_covariance_buffer, cross_covariance_values,
				sizeof(cross_covariance_values));
		init_raw_buffer(&corrector_covariance_buffer,
				corrector_covariance_values,
				sizeof(corrector_covariance_values));
		init_raw_buffer(&sample_count_buffer, sample_count_value,
				sizeof(sample_count_value));
		init_raw_buffer(&matrix_buffer, matrix_values, sizeof(matrix_values));
		init_raw_buffer(&status_buffer, status_value, sizeof(status_value));
		init_raw_buffer(&status_coordinate_buffer, status_coordinate_values,
				sizeof(status_coordinate_values));
		init_raw_buffer(&diagnostic_sample_count_buffer,
				diagnostic_sample_count_value,
				sizeof(diagnostic_sample_count_value));
		init_raw_buffer(&variance_range_buffer, variance_range_values,
				sizeof(variance_range_values));
		init_raw_buffer(&correlation_buffer, correlation_value,
				sizeof(correlation_value));
		init_raw_buffer(&pivot_ratio_buffer, pivot_ratio_value,
				sizeof(pivot_ratio_value));
		cross_covariance_buffer.header.seq = 92;
		open_loop_inputs[0] = &cross_covariance_buffer.buffer;
		open_loop_inputs[1] = &corrector_covariance_buffer.buffer;
		open_loop_inputs[2] = &sample_count_buffer.buffer;
		open_loop_outputs[0] = &matrix_buffer.buffer;
		open_loop_outputs[1] = &status_buffer.buffer;
		open_loop_outputs[2] = &status_coordinate_buffer.buffer;
		open_loop_outputs[3] = &diagnostic_sample_count_buffer.buffer;
		open_loop_outputs[4] = &variance_range_buffer.buffer;
		open_loop_outputs[5] = &correlation_buffer.buffer;
		open_loop_outputs[6] = &pivot_ratio_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, open_loop_inputs, 3,
				open_loop_outputs, 7) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(fabsf(matrix_values[0] - 2.0f) < 1.0e-6f &&
				fabsf(matrix_values[1] + 1.0f) < 1.0e-6f &&
				fabsf(matrix_values[2] - 0.5f) < 1.0e-6f &&
				fabsf(matrix_values[3] - 3.0f) < 1.0e-6f);
		assert(status_value[0] == 0 && status_coordinate_values[0] == 0 &&
				status_coordinate_values[1] == 0);
		assert(diagnostic_sample_count_value[0] == 3);
		assert(variance_range_values[0] == 2.0f &&
				variance_range_values[1] == 2.0f);
		assert(fabsf(correlation_value[0] - 0.5f) < 1.0e-6f);
		assert(fabsf(pivot_ratio_value[0] - 0.75f) < 1.0e-6f);
		for (size_t i = 0; i < 7; i++) {
			struct spa_meta_header *header = open_loop_outputs[i]->metas[0].data;
			assert(header->seq == 92);
		}
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = closed-loop-solve plugin = \"%s\""
		"   label = docrime-closed-loop-interaction-matrix-f32"
		" config = { measurement_count = 2 corrector_coordinate_count = 2"
		"              minimum_corrector_variance = 0.1"
		"              cross_covariance_schema = test.filtergraph/test-cross-covariance/1"
		"              corrector_covariance_schema = test.filtergraph/test-corrector-covariance/1"
		"              interaction_matrix_schema = test.filtergraph/test-interaction-matrix/1"
		"              rate = [ 1 1 ] }"
		"   props = { \"loop-gain\" = 0.4"
		"             \"maximum-absolute-corrector-correlation\" = 0.8"
		"             \"minimum-factorization-pivot-ratio\" = 0.02 } }"
		"] inputs = [ \"closed-loop-solve:measurement-corrector-cross-covariance-sum\""
		"             \"closed-loop-solve:corrector-covariance-sum\""
		"             \"closed-loop-solve:sample-count\" ]"
		" outputs = [ \"closed-loop-solve:interaction-matrix\""
		"             \"closed-loop-solve:status\""
		"             \"closed-loop-solve:status-coordinates\""
		"             \"closed-loop-solve:diagnostic-sample-count\""
		"             \"closed-loop-solve:corrector-variance-range\""
		"             \"closed-loop-solve:maximum-absolute-corrector-correlation\""
		"             \"closed-loop-solve:minimum-factorization-pivot-ratio\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "DO-CRIME closed-loop solve graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 3);
	assert(spa_fgn_graph_get_n_outputs(graph) == 7);

	{
		float cross_covariance_values[4] = { 6.0f, 0.0f, 8.0f, 13.0f };
		float corrector_covariance_values[4] = { 4.0f, 2.0f, 2.0f, 4.0f };
		uint64_t sample_count_value[1] = { 3 };
		float matrix_values[4] = { 0.0f };
		uint8_t status_value[1] = { UINT8_MAX };
		uint64_t status_coordinate_values[2] = { UINT64_MAX, UINT64_MAX };
		uint64_t diagnostic_sample_count_value[1] = { UINT64_MAX };
		float variance_range_values[2] = { 0.0f };
		float correlation_value[1] = { 0.0f };
		float pivot_ratio_value[1] = { 0.0f };
		struct raw_test_buffer cross_covariance_buffer;
		struct raw_test_buffer corrector_covariance_buffer, sample_count_buffer;
		struct raw_test_buffer matrix_buffer, status_buffer;
		struct raw_test_buffer status_coordinate_buffer;
		struct raw_test_buffer diagnostic_sample_count_buffer;
		struct raw_test_buffer variance_range_buffer, correlation_buffer;
		struct raw_test_buffer pivot_ratio_buffer;
		struct spa_buffer *closed_loop_inputs[3];
		struct spa_buffer *closed_loop_outputs[7];

		init_raw_buffer(&cross_covariance_buffer, cross_covariance_values,
				sizeof(cross_covariance_values));
		init_raw_buffer(&corrector_covariance_buffer,
				corrector_covariance_values,
				sizeof(corrector_covariance_values));
		init_raw_buffer(&sample_count_buffer, sample_count_value,
				sizeof(sample_count_value));
		init_raw_buffer(&matrix_buffer, matrix_values, sizeof(matrix_values));
		init_raw_buffer(&status_buffer, status_value, sizeof(status_value));
		init_raw_buffer(&status_coordinate_buffer, status_coordinate_values,
				sizeof(status_coordinate_values));
		init_raw_buffer(&diagnostic_sample_count_buffer,
				diagnostic_sample_count_value,
				sizeof(diagnostic_sample_count_value));
		init_raw_buffer(&variance_range_buffer, variance_range_values,
				sizeof(variance_range_values));
		init_raw_buffer(&correlation_buffer, correlation_value,
				sizeof(correlation_value));
		init_raw_buffer(&pivot_ratio_buffer, pivot_ratio_value,
				sizeof(pivot_ratio_value));
		cross_covariance_buffer.header.seq = 93;
		closed_loop_inputs[0] = &cross_covariance_buffer.buffer;
		closed_loop_inputs[1] = &corrector_covariance_buffer.buffer;
		closed_loop_inputs[2] = &sample_count_buffer.buffer;
		closed_loop_outputs[0] = &matrix_buffer.buffer;
		closed_loop_outputs[1] = &status_buffer.buffer;
		closed_loop_outputs[2] = &status_coordinate_buffer.buffer;
		closed_loop_outputs[3] = &diagnostic_sample_count_buffer.buffer;
		closed_loop_outputs[4] = &variance_range_buffer.buffer;
		closed_loop_outputs[5] = &correlation_buffer.buffer;
		closed_loop_outputs[6] = &pivot_ratio_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, closed_loop_inputs, 3,
				closed_loop_outputs, 7) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(fabsf(matrix_values[0] - 0.8f) < 2.0e-6f &&
				fabsf(matrix_values[1] + 0.4f) < 2.0e-6f &&
				fabsf(matrix_values[2] - 0.2f) < 2.0e-6f &&
				fabsf(matrix_values[3] - 1.2f) < 2.0e-6f);
		assert(status_value[0] == 0 && diagnostic_sample_count_value[0] == 3);
		assert(matrix_buffer.header.seq == 93 && status_buffer.header.seq == 93);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = pseudo-open-loop plugin = \"%s\""
		"   label = pseudo-open-loop-reconstruction-f32"
		" config = { measurement_count = 2 corrector_shape_count = 1"
		"              residual_error_count = 1"
		"              residual_measurements_schema = pipewireao/shack-hartmann-slopes-f32/1"
		"              corrector_shape_schema = pipewireao/requested-pdm-command-f32/1"
		"              pseudo_open_loop_measurements_schema = pipewireao/shack-hartmann-slopes-f32/1"
		"              residual_error_schema = pipewireao/controller-residual-error-f32/1"
		"              rate = [ 1000 1 ]  initial_interaction_matrix = null  initial_reconstructor = null  initial_corrector_to_error = null } }"
		"] inputs = [ \"pseudo-open-loop:residual-measurements\""
		"             \"pseudo-open-loop:corrector-shape\""
		"             \"pseudo-open-loop:interaction-matrix\""
		"             \"pseudo-open-loop:reconstructor\""
		"             \"pseudo-open-loop:corrector-to-error\" ]"
		" outputs = [ \"pseudo-open-loop:pseudo-open-loop-measurements\""
		"             \"pseudo-open-loop:residual-error\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "pseudo-open-loop declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 5);
	assert(spa_fgn_graph_get_n_outputs(graph) == 2);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			2, &format) == 0);
	assert(format->n_dimensions == 2 && format->shape[0] == 2 &&
			format->shape[1] == 1);
	assert(strcmp(format->schema,
			"pipewireao/pseudo-open-loop-interaction-matrix-f32/1") == 0);

	{
		float residual_measurement_values[2] = { 4.0f, 3.0f };
		float corrector_shape_values[1] = { 1.0f };
		float pseudo_open_loop_values[2] = { 0.0f };
		float residual_error_values[1] = { 0.0f };
		float interaction_values[2] = { 2.0f, 1.0f };
		float reconstructor_values[2] = { 0.5f, 1.0f };
		float corrector_to_error_values[1] = { 2.0f };
		struct raw_test_buffer residual_measurement_buffer;
		struct raw_test_buffer corrector_shape_buffer;
		struct raw_test_buffer pseudo_open_loop_buffer;
		struct raw_test_buffer residual_error_buffer;
		struct raw_test_buffer interaction_buffer, reconstructor_buffer;
		struct raw_test_buffer corrector_to_error_buffer;
		struct spa_buffer *pseudo_open_loop_inputs[5];
		struct spa_buffer *pseudo_open_loop_outputs[2];

		init_raw_buffer(&residual_measurement_buffer,
				residual_measurement_values,
				sizeof(residual_measurement_values));
		init_raw_buffer(&corrector_shape_buffer, corrector_shape_values,
				sizeof(corrector_shape_values));
		init_raw_buffer(&pseudo_open_loop_buffer, pseudo_open_loop_values,
				sizeof(pseudo_open_loop_values));
		init_raw_buffer(&residual_error_buffer, residual_error_values,
				sizeof(residual_error_values));
		init_raw_buffer(&interaction_buffer, interaction_values,
				sizeof(interaction_values));
		init_raw_buffer(&reconstructor_buffer, reconstructor_values,
				sizeof(reconstructor_values));
		init_raw_buffer(&corrector_to_error_buffer, corrector_to_error_values,
				sizeof(corrector_to_error_values));
		residual_measurement_buffer.header.seq = 43;
		pseudo_open_loop_inputs[0] = &residual_measurement_buffer.buffer;
		pseudo_open_loop_inputs[1] = &corrector_shape_buffer.buffer;
		pseudo_open_loop_inputs[2] = NULL;
		pseudo_open_loop_inputs[3] = NULL;
		pseudo_open_loop_inputs[4] = NULL;
		pseudo_open_loop_outputs[0] = &pseudo_open_loop_buffer.buffer;
		pseudo_open_loop_outputs[1] = &residual_error_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, pseudo_open_loop_inputs, 5,
				pseudo_open_loop_outputs, 2) == 0);
		assert(pseudo_open_loop_values[0] == 4.0f &&
				pseudo_open_loop_values[1] == 3.0f);
		assert(residual_error_values[0] == 0.0f);

		assert(spa_fgn_graph_update_parameter(graph, 2,
				&interaction_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, pseudo_open_loop_inputs, 5,
				pseudo_open_loop_outputs, 2) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(pseudo_open_loop_values[0] == 6.0f &&
				pseudo_open_loop_values[1] == 4.0f);
		assert(residual_error_values[0] == 0.0f);

		assert(spa_fgn_graph_update_parameter(graph, 3,
				&reconstructor_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, pseudo_open_loop_inputs, 5,
				pseudo_open_loop_outputs, 2) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(residual_error_values[0] == 7.0f);

		assert(spa_fgn_graph_update_parameter(graph, 4,
				&corrector_to_error_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, pseudo_open_loop_inputs, 5,
				pseudo_open_loop_outputs, 2) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(residual_error_values[0] == 5.0f);
		assert(pseudo_open_loop_buffer.header.seq == 43 &&
				residual_error_buffer.header.seq == 43);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = project plugin = \"%s\""
		"   label = controller-to-vdm-f32"
		" config = { controller_command_count = 277 active_vdm_count = 277"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = decompose plugin = \"%s\""
		"   label = vdm-to-pdm-f32"
		" config = { active_vdm_count = 277 full_vdm_count = 277"
		"              physical_actuator_count = 277"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = command plugin = \"%s\""
		"   label = pdm-command-f32"
		" config = { actuator_count = 277 command_limits = [ -1.0"
		" 1.0 ]               rate = [ 1000 1 ]  slew_limit = null  initial_current = null  quantization_origin = null  quantization_step = null  dither_seed = null } }"
		" { type = ndarray name = feedback plugin = \"%s\""
		"   label = pdm-feedback-to-vdm-f32"
		" config = { active_vdm_count = 277 full_vdm_count = 277"
		"              physical_actuator_count = 277"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = controller-feedback plugin = \"%s\""
		"   label = vdm-feedback-to-controller-f32"
		" config = { controller_command_count = 277 active_vdm_count = 277"
		"              rate = [ 1000 1 ] } }"
		"] links = ["
		" { output = \"project:vdm-command\""
		"   input = \"decompose:vdm-command\" }"
		" { output = \"decompose:requested-pdm-command\""
		"   input = \"command:requested\" }"
		" { output = \"command:constraint-feedback\""
		"   input = \"feedback:pdm-constraint-feedback\" }"
		" { output = \"feedback:vdm-constraint-feedback\""
		"   input = \"controller-feedback:vdm-constraint-feedback\" }"
		"] inputs = [ \"project:controller-command\""
		"             \"project:controller-to-vdm\""
		"             \"decompose:active-to-full\""
		"             \"decompose:vdm-to-pdm\""
		"             \"feedback:full-to-active\""
		"             \"feedback:pdm-to-vdm\""
		"             \"controller-feedback:vdm-to-controller\" ]"
		" outputs = [ \"command:demanded\""
		"             \"controller-feedback:controller-constraint-feedback\" ] }",
		argv[1], argv[1], argv[1], argv[1], argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "decomposed DM declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 7);
	assert(spa_fgn_graph_get_n_outputs(graph) == 2);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			2, &format) == 0);
	assert(format->n_dimensions == 2 &&
			format->shape[0] == TEST_ACTUATORS &&
			format->shape[1] == TEST_ACTUATORS);
	assert(strcmp(format->schema,
			"pipewireao/active-to-full-vdm-projection-f32/1") == 0);

	{
		float controller_command_values[TEST_ACTUATORS] = { 0.0f };
		float demanded_values[TEST_ACTUATORS] = { 0.0f };
		float controller_feedback_values[TEST_ACTUATORS] = { 0.0f };
		float identity_values[TEST_ACTUATORS * TEST_ACTUATORS] = { 0.0f };
		struct raw_test_buffer controller_command_buffer, demanded_buffer;
		struct raw_test_buffer controller_feedback_buffer, identity_buffer;
		struct spa_buffer *dm_inputs[7];
		struct spa_buffer *dm_outputs[2];

		controller_command_values[0] = 2.0f;
		controller_command_values[TEST_ACTUATORS - 1] = -0.5f;
		for (size_t i = 0; i < TEST_ACTUATORS; i++)
			identity_values[i * TEST_ACTUATORS + i] = 1.0f;
		init_raw_buffer(&controller_command_buffer, controller_command_values,
				sizeof(controller_command_values));
		init_raw_buffer(&demanded_buffer, demanded_values,
				sizeof(demanded_values));
		init_raw_buffer(&controller_feedback_buffer, controller_feedback_values,
				sizeof(controller_feedback_values));
		init_raw_buffer(&identity_buffer, identity_values,
				sizeof(identity_values));
		controller_command_buffer.header.seq = 91;
		dm_inputs[0] = &controller_command_buffer.buffer;
		for (size_t i = 1; i < 7; i++)
			dm_inputs[i] = NULL;
		dm_outputs[0] = &demanded_buffer.buffer;
		dm_outputs[1] = &controller_feedback_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, dm_inputs, 7,
				dm_outputs, 2) == 0);
		for (uint32_t port = 1; port < 7; port++) {
			identity_buffer.header.seq = 100 + port;
			assert(spa_fgn_graph_update_parameter(graph, port,
					&identity_buffer.buffer) == 0);
			assert(spa_fgn_graph_process(graph, dm_inputs, 7,
					dm_outputs, 2) ==
					SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		}
		assert(demanded_values[0] == 1.0f &&
				demanded_values[TEST_ACTUATORS - 1] == -0.5f);
		assert(controller_feedback_values[0] == 1.0f &&
				controller_feedback_values[TEST_ACTUATORS - 1] == 0.0f);
		for (size_t i = 1; i + 1 < TEST_ACTUATORS; i++) {
			assert(demanded_values[i] == 0.0f);
			assert(controller_feedback_values[i] == 0.0f);
		}
		assert(demanded_buffer.header.seq == 91 &&
				controller_feedback_buffer.header.seq == 91);
		assert(spa_fgn_graph_process(graph, dm_inputs, 7,
				dm_outputs, 2) == 0);
		assert(demanded_values[0] == 1.0f &&
				demanded_values[TEST_ACTUATORS - 1] == -0.5f);
		assert(controller_feedback_values[0] == 1.0f &&
				controller_feedback_values[TEST_ACTUATORS - 1] == 0.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = controller plugin = \"%s\""
		"   label = closed-loop-correction-f32"
		" config = { state_length = 2 hidden_mode_count = 1"
		"              rate = [ 1000 1 ] }"
		"   props = { gain = 0.5 pole = 0.75"
		"             \"anti-windup-gain\" = 0.8"
		"             \"hidden-mode-gain\" = 0.25 } }"
		"] inputs = [ \"controller:residual-error\""
		"             \"controller:constraint-feedback\""
		"             \"controller:shape-to-hidden\""
		"             \"controller:hidden-to-shape\" ]"
		" outputs = [ \"controller:correction\""
		"             \"controller:controller-state\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "closed-loop declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 4);
	assert(spa_fgn_graph_get_n_outputs(graph) == 2);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			2, &format) == 0);
	assert(format->n_dimensions == 2 && format->shape[0] == 1 &&
			format->shape[1] == 2);
	assert(format->rate_num == 1000 && format->rate_denom == 1);

	{
		float residual_values[2] = { 2.0f, -2.0f };
		float feedback_values[2] = { 99.0f, 99.0f };
		float correction_values[2] = { 0.0f };
		float state_values[2] = { 0.0f };
		float shape_to_hidden_values[2] = { 1.0f, 1.0f };
		float hidden_to_shape_values[2] = { 0.5f, 0.5f };
		struct raw_test_buffer residual_buffer, feedback_buffer;
		struct raw_test_buffer correction_buffer, state_buffer;
		struct raw_test_buffer shape_to_hidden_buffer, hidden_to_shape_buffer;
		struct spa_buffer *controller_inputs[4];
		struct spa_buffer *controller_outputs[2];

		init_raw_buffer(&residual_buffer, residual_values,
				sizeof(residual_values));
		init_raw_buffer(&feedback_buffer, feedback_values,
				sizeof(feedback_values));
		init_raw_buffer(&correction_buffer, correction_values,
				sizeof(correction_values));
		init_raw_buffer(&state_buffer, state_values, sizeof(state_values));
		init_raw_buffer(&shape_to_hidden_buffer, shape_to_hidden_values,
				sizeof(shape_to_hidden_values));
		init_raw_buffer(&hidden_to_shape_buffer, hidden_to_shape_values,
				sizeof(hidden_to_shape_values));
		residual_buffer.header.seq = 61;
		shape_to_hidden_buffer.header.seq = 11;
		hidden_to_shape_buffer.header.seq = 12;
		controller_inputs[0] = &residual_buffer.buffer;
		controller_inputs[1] = &feedback_buffer.buffer;
		controller_inputs[2] = NULL;
		controller_inputs[3] = NULL;
		controller_outputs[0] = &correction_buffer.buffer;
		controller_outputs[1] = &state_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, controller_inputs, 4,
				controller_outputs, 2) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(correction_values[0] == 1.0f && correction_values[1] == -1.0f);
		assert(state_values[0] == 0.0f && state_values[1] == 0.0f);
		assert(correction_buffer.header.seq == 61 && state_buffer.header.seq == 61);

		assert(spa_fgn_graph_update_parameter(graph, 2,
				&shape_to_hidden_buffer.buffer) == 0);
		residual_values[0] = 0.0f;
		residual_values[1] = 0.0f;
		feedback_values[0] = 0.2f;
		feedback_values[1] = -0.4f;
		assert(spa_fgn_graph_process(graph, controller_inputs, 4,
				controller_outputs, 2) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(fabsf(correction_values[0] - 0.63f) < 1.0e-6f);
		assert(fabsf(correction_values[1] + 0.51f) < 1.0e-6f);
		assert(fabsf(state_values[0] - 0.84f) < 1.0e-6f);
		assert(fabsf(state_values[1] + 0.68f) < 1.0e-6f);

		assert(spa_fgn_graph_update_parameter(graph, 3,
				&hidden_to_shape_buffer.buffer) == 0);
		feedback_values[0] = 0.0f;
		feedback_values[1] = 0.0f;
		assert(spa_fgn_graph_process(graph, controller_inputs, 4,
				controller_outputs, 2) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(fabsf(correction_values[0] - 0.46125f) < 1.0e-6f);
		assert(fabsf(correction_values[1] + 0.39375f) < 1.0e-6f);
		assert(fabsf(state_values[0] - 0.615f) < 1.0e-6f);
		assert(fabsf(state_values[1] + 0.525f) < 1.0e-6f);

		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, controller_inputs, 4,
				controller_outputs, 2) == 0);
		assert(correction_values[0] == 0.0f && correction_values[1] == 0.0f);
		assert(state_values[0] == 0.0f && state_values[1] == 0.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ workers = { helpers = 3 } nodes = ["
		" { type = ndarray name = measure plugin = \"%s\""
		"   label = shack-hartmann-image-f32"
		" config = { image_size = [ 2 2 ] "
		"              subaperture_size = [ 1 2 ] "
		"              subaperture_count = 2"
		"              image_schema = pipewireao/calibrated-pixels-f32/1"
		"              coordinate_scale = 1.0 pixel_threshold = 0.0"
		"              flux_threshold = 0.0 rate = [ 1000 1 ]  initial_subaperture_origins = null  reference_slopes = null  active = null } }"
		" { type = ndarray name = reconstruct plugin = \"%s\""
		"   label = shwfs-reconstructor-f32"
		" config = { reconstructed_count = 1 subaperture_count = 2"
		"              initial_reconstructor = [ 1.0 0.0 0.0 0.0 ]"
		"              reconstructed_schema = pipewireao/controller-residual-error-f32/1"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = integrate plugin = \"%s\""
		"   label = leaky-integrator-f32"
		" config = { extent = 1 initial_state = 0.0"
		"              input_schema = pipewireao/controller-residual-error-f32/1"
		"              output_schema = pipewireao/requested-pdm-command-f32/1"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = command plugin = \"%s\""
		"   label = pdm-command-f32"
		" config = { actuator_count = 1 command_limits = [ -1.0"
		" 1.0 ]               rate = [ 1000 1 ]  slew_limit = null  initial_current = null  quantization_origin = null  quantization_step = null  dither_seed = null } }"
		"] links = ["
		" { output = \"measure:slopes\" input = \"reconstruct:slopes\" }"
		" { output = \"reconstruct:reconstructed\" input = \"integrate:input\" }"
		" { output = \"integrate:output\" input = \"command:requested\" }"
		"] inputs = [ \"measure:image\" \"measure:subaperture-origins\""
		"             \"measure:coordinates\" \"measure:reference-slopes\""
		"             \"measure:thresholds\" \"measure:active\""
		"             \"reconstruct:reconstructor\" ]"
		" outputs = [ \"command:demanded\" \"measure:flux\""
		"             \"measure:validity\" ] }",
		argv[1], argv[1], argv[1], argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "Shack-Hartmann declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 7);
	assert(spa_fgn_graph_get_n_outputs(graph) == 3);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->n_dimensions == 2 && format->shape[0] == 2 &&
			format->shape[1] == 2);
	assert(format->rate_num == 1000 && format->rate_denom == 1);

	{
		float image_values[4] = { 1.0f, 3.0f, 2.0f, 2.0f };
		float reconstructed_values[1] = { 0.0f };
		float flux_values[2] = { 0.0f };
		uint8_t validity_values[2] = { 0 };
		uint8_t active_values[2] = { 0, 1 };
		struct raw_test_buffer image_buffer, reconstructed_buffer, flux_buffer;
		struct raw_test_buffer validity_buffer, active_buffer;
		struct spa_buffer *image_inputs[7];
		struct spa_buffer *measurement_outputs[3];

		init_raw_buffer(&image_buffer, image_values, sizeof(image_values));
		init_raw_buffer(&reconstructed_buffer, reconstructed_values,
				sizeof(reconstructed_values));
		init_raw_buffer(&flux_buffer, flux_values, sizeof(flux_values));
		init_raw_buffer(&validity_buffer, validity_values,
				sizeof(validity_values));
		init_raw_buffer(&active_buffer, active_values, sizeof(active_values));
		image_buffer.header.seq = 52;
		active_buffer.header.seq = 17;
		image_inputs[0] = &image_buffer.buffer;
		for (size_t i = 1; i < 7; i++)
			image_inputs[i] = NULL;
		measurement_outputs[0] = &reconstructed_buffer.buffer;
		measurement_outputs[1] = &flux_buffer.buffer;
		measurement_outputs[2] = &validity_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, image_inputs, 7,
				measurement_outputs, 3) == 0);
		assert(reconstructed_values[0] == 0.25f);
		assert(flux_values[0] == 4.0f && flux_values[1] == 4.0f);
		assert(validity_values[0] == 1 && validity_values[1] == 1);
		assert(reconstructed_buffer.header.seq == 52 && flux_buffer.header.seq == 52 &&
				validity_buffer.header.seq == 52);

		assert(spa_fgn_graph_update_parameter(graph, 5,
				&active_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, image_inputs, 7,
				measurement_outputs, 3) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(reconstructed_values[0] == 0.0f);
		assert(validity_values[0] == 1);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ workers = { helpers = 3 } nodes = ["
		" { type = ndarray name = normalize plugin = \"%s\""
		"   label = pyramid-pupil-image-f32"
		" config = { image_size = [ 4 4 ] "
		"              pupil_size = [ 2 2 ] "
		"              pupil_origins = [ [ 0 0 ] [ 0 2 ] [ 2 0 ] [ 2 2 ] ]"
		"              pupil_mask = [ true false true true ]"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = reconstruct plugin = \"%s\""
		"   label = pwfs-pupil-reconstructor-f32"
		" config = { reconstructed_count = 2 selected_pixel_count = 3"
		"              initial_reconstructor = ["
		"                1 0 0 0 0 0 0 0 0 0 0 0"
		"                0 1 0 0 0 0 0 0 0 0 0 0 ]"
		"              reconstructed_schema = pipewireao/controller-residual-error-f32/1"
		"              rate = [ 1000 1 ] } }"
		"] inputs = [ \"normalize:image\" \"normalize:pupil-origins\""
		"             \"normalize:pupil-mask\" \"reconstruct:reconstructor\" ]"
		" links = [ { output = \"normalize:reconstruction-pixels\""
		"             input = \"reconstruct:reconstruction-pixels\" } ]"
		" outputs = [ \"normalize:reconstruction-pixels\""
		"             \"normalize:mean-pupil-intensity\""
		"             \"reconstruct:reconstructed\" ] }",
		argv[1], argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "pyramid declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 4);
	assert(spa_fgn_graph_get_n_outputs(graph) == 3);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			0, &format) == 0);
	assert(format->n_dimensions == 2 && format->shape[0] == 4 &&
			format->shape[1] == 3);
	assert(format->layout == SPA_NDARRAY_LAYOUT_ROW_MAJOR);
	assert(format->rate_num == 1000 && format->rate_denom == 1);

	{
		float image_values[16];
		float pixel_values[12] = { 0.0f };
		float mean_value[1] = { 0.0f };
		float reconstructed_values[2] = { 0.0f };
		uint8_t mask_values[4] = { 0, 1, 1, 1 };
		struct raw_test_buffer image_buffer, pixel_buffer, mean_buffer;
		struct raw_test_buffer reconstructed_buffer, mask_buffer;
		struct spa_buffer *pyramid_inputs[4];
		struct spa_buffer *pyramid_outputs[3];

		for (size_t i = 0; i < 16; i++)
			image_values[i] = (float)i + 1.0f;
		init_raw_buffer(&image_buffer, image_values, sizeof(image_values));
		init_raw_buffer(&pixel_buffer, pixel_values, sizeof(pixel_values));
		init_raw_buffer(&mean_buffer, mean_value, sizeof(mean_value));
		init_raw_buffer(&reconstructed_buffer, reconstructed_values,
				sizeof(reconstructed_values));
		init_raw_buffer(&mask_buffer, mask_values, sizeof(mask_values));
		image_buffer.header.seq = 71;
		mask_buffer.header.seq = 33;
		pyramid_inputs[0] = &image_buffer.buffer;
		pyramid_inputs[1] = NULL;
		pyramid_inputs[2] = NULL;
		pyramid_inputs[3] = NULL;
		pyramid_outputs[0] = &pixel_buffer.buffer;
		pyramid_outputs[1] = &mean_buffer.buffer;
		pyramid_outputs[2] = &reconstructed_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, pyramid_inputs, 4,
				pyramid_outputs, 3) == 0);
		assert(mean_value[0] == 8.5f && pixel_values[0] == 0.0f);
		assert(reconstructed_values[0] == 0.0f && reconstructed_values[1] == 0.0f);
		assert(pixel_buffer.header.seq == 71 && mean_buffer.header.seq == 71 &&
				reconstructed_buffer.header.seq == 71);
		assert(spa_fgn_graph_process(graph, pyramid_inputs, 4,
				pyramid_outputs, 3) == 0);
		assert(pixel_values[0] == 1.0f / 8.5f);
		assert(pixel_values[1] == 5.0f / 8.5f);
		assert(pixel_values[2] == 6.0f / 8.5f);
		assert(reconstructed_values[0] == pixel_values[0]);
		assert(reconstructed_values[1] == pixel_values[1]);

		assert(spa_fgn_graph_update_parameter(graph, 2,
				&mask_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, pyramid_inputs, 4,
				pyramid_outputs, 3) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(pixel_values[0] == 2.0f / 8.5f);
		assert(pixel_values[1] == 5.0f / 8.5f);
		assert(pixel_values[2] == 6.0f / 8.5f);
		assert(reconstructed_values[0] == pixel_values[0]);
		assert(reconstructed_values[1] == pixel_values[1]);
		assert(spa_fgn_graph_reset(graph) == 0);
		assert(spa_fgn_graph_process(graph, pyramid_inputs, 4,
				pyramid_outputs, 3) == 0);
		assert(pixel_values[0] == 0.0f);
		assert(reconstructed_values[0] == 0.0f && reconstructed_values[1] == 0.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = chopped plugin = \"%s\""
		"   label = chopped-image-difference-f32"
		" config = { pixel_count = 2 trigger_high_is_fringed = true rate = [ 1000 1 ] } }"
		"] inputs = [ \"chopped:image\" \"chopped:input-validity\""
		"              ]"
		" outputs = [ \"chopped:difference\""
		"             \"chopped:output-validity\""
		"             \"chopped:pair-status\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "chopped-image declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 3);


	{
		float image_values[2] = { 10.0f, 20.0f };
		uint8_t input_validity_values[2] = { 1, 1 };
		struct spa_meta_electrical_trigger trigger = { .flags = SPA_META_RECORD_VALID, .classification = SPA_META_TRIGGER_HIGH };
		struct spa_meta_electrical_trigger output_triggers[3] = {0};
		struct spa_meta output_metas[3][2];
		struct spa_meta image_metas[2];
		float difference_values[2] = { -1.0f, -1.0f };
		uint8_t output_validity_values[2] = { 1, 1 };
		uint8_t status_value[1] = { UINT8_MAX };
		struct raw_test_buffer image_buffer, input_validity_buffer;
		struct raw_test_buffer difference_buffer, output_validity_buffer;
		struct raw_test_buffer status_buffer;
		struct spa_buffer *chopped_inputs[2];
		struct spa_buffer *chopped_outputs[3];

		init_raw_buffer(&image_buffer, image_values, sizeof(image_values));
		init_raw_buffer(&input_validity_buffer, input_validity_values,
				sizeof(input_validity_values));
		trigger.classification = SPA_META_TRIGGER_HIGH;
		image_metas[0] = image_buffer.meta;
		image_metas[1] = (struct spa_meta){ .type = SPA_META_ElectricalTrigger, .size = sizeof(trigger), .data = &trigger };
		image_buffer.buffer.metas = image_metas;
		image_buffer.buffer.n_metas = 2;
		init_raw_buffer(&difference_buffer, difference_values,
				sizeof(difference_values));
		init_raw_buffer(&output_validity_buffer, output_validity_values,
				sizeof(output_validity_values));
		init_raw_buffer(&status_buffer, status_value, sizeof(status_value));
		image_buffer.header.seq = 91;
		chopped_inputs[0] = &image_buffer.buffer;
		chopped_inputs[1] = &input_validity_buffer.buffer;
		chopped_outputs[0] = &difference_buffer.buffer;
		chopped_outputs[1] = &output_validity_buffer.buffer;
		chopped_outputs[2] = &status_buffer.buffer;

		for (uint32_t i = 0; i < 3; i++) {
			output_metas[i][0] = chopped_outputs[i]->metas[0];
			output_metas[i][1] = (struct spa_meta){ .type = SPA_META_ElectricalTrigger, .size = sizeof(output_triggers[i]), .data = &output_triggers[i] };
			chopped_outputs[i]->metas = output_metas[i];
			chopped_outputs[i]->n_metas = 2;
		}

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, chopped_inputs, 2,
				chopped_outputs, 3) == 0);
		assert(difference_values[0] == 0.0f && difference_values[1] == 0.0f);
		assert(output_validity_values[0] == 0 &&
				output_validity_values[1] == 0 && status_value[0] == 0);
		assert(difference_buffer.header.seq == 91 &&
				output_validity_buffer.header.seq == 91 &&
				status_buffer.header.seq == 91);

		image_values[0] = 3.0f;
		image_values[1] = 8.0f;
		trigger.classification = SPA_META_TRIGGER_LOW;
		image_buffer.header.seq = 92;
		assert(spa_fgn_graph_process(graph, chopped_inputs, 2,
				chopped_outputs, 3) == 0);
		assert(difference_values[0] == 7.0f && difference_values[1] == 12.0f);
		assert(output_validity_values[0] == 1 &&
				output_validity_values[1] == 1 && status_value[0] == 1);
		assert(difference_buffer.header.seq == 92 &&
				output_validity_buffer.header.seq == 92 &&
				status_buffer.header.seq == 92);

		trigger.classification = SPA_META_TRIGGER_MIXED;
		assert(spa_fgn_graph_process(graph, chopped_inputs, 2,
				chopped_outputs, 3) == -EINVAL);
		trigger.classification = SPA_META_TRIGGER_HIGH;
		image_values[0] = 20.0f;
		image_values[1] = 30.0f;
		assert(spa_fgn_graph_process(graph, chopped_inputs, 2,
				chopped_outputs, 3) == 0);
		assert(difference_values[0] == 17.0f && difference_values[1] == 22.0f);
		assert(status_value[0] == 1);

		assert(spa_fgn_graph_reset(graph) == 0);
		trigger.classification = SPA_META_TRIGGER_LOW;
		assert(spa_fgn_graph_process(graph, chopped_inputs, 2,
				chopped_outputs, 3) == 0);
		assert(difference_values[0] == 0.0f && difference_values[1] == 0.0f);
		assert(output_validity_values[0] == 0 &&
				output_validity_values[1] == 0 && status_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = centroid plugin = \"%s\""
		"   label = center-of-gravity-f32"
		" config = { pixel_count = 3"
		"              initial_coordinates = [ [ -1.0 0.0 ] [ 0.0 1.0 ]"
		"                                      [ 1.0 2.0 ] ]"
		"              rate = [ 1000 1 ] }"
		"   props = { \"reference-centroid-x\" = 0.25"
		"             \"reference-centroid-y\" = -0.5"
		"             \"flux-threshold\" = 4.0 } }"
		"] inputs = [ \"centroid:pixels\" \"centroid:coordinates\" ]"
		" outputs = [ \"centroid:centroid-offset\""
		"             \"centroid:flux\" \"centroid:validity\" ] }",
		argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "center-of-gravity declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 2);
	assert(spa_fgn_graph_get_n_outputs(graph) == 3);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			1, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE &&
			format->n_dimensions == 2 && format->shape[0] == 3 &&
			format->shape[1] == 2);
	assert(strcmp(format->schema,
			"pipewireao/center-of-gravity-coordinates-f32/1") == 0);

	{
		float pixel_values[3] = { 0.0f, 0.0f, 4.0f };
		float coordinate_values[6] = { -1.0f, 0.0f, 0.0f, 1.0f,
				1.0f, 2.0f };
		float offset_values[2] = { 0.0f };
		float flux_value[1] = { 0.0f };
		uint8_t validity_value[1] = { 0 };
		struct raw_test_buffer pixel_buffer, coordinate_buffer, offset_buffer;
		struct raw_test_buffer flux_buffer, validity_buffer;
		struct spa_buffer *centroid_inputs[2];
		struct spa_buffer *centroid_outputs[3];

		init_raw_buffer(&pixel_buffer, pixel_values, sizeof(pixel_values));
		init_raw_buffer(&coordinate_buffer, coordinate_values,
				sizeof(coordinate_values));
		init_raw_buffer(&offset_buffer, offset_values, sizeof(offset_values));
		init_raw_buffer(&flux_buffer, flux_value, sizeof(flux_value));
		init_raw_buffer(&validity_buffer, validity_value,
				sizeof(validity_value));
		pixel_buffer.header.seq = 101;
		centroid_inputs[0] = &pixel_buffer.buffer;
		centroid_inputs[1] = NULL;
		centroid_outputs[0] = &offset_buffer.buffer;
		centroid_outputs[1] = &flux_buffer.buffer;
		centroid_outputs[2] = &validity_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, centroid_inputs, 2,
				centroid_outputs, 3) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(offset_values[0] == 0.75f && offset_values[1] == 2.5f);
		assert(flux_value[0] == 4.0f && validity_value[0] == 1);
		assert(offset_buffer.header.seq == 101 && flux_buffer.header.seq == 101 &&
				validity_buffer.header.seq == 101);

		coordinate_values[0] = -2.0f;
		coordinate_values[3] = 2.0f;
		coordinate_values[4] = 2.0f;
		coordinate_values[5] = 4.0f;
		assert(spa_fgn_graph_update_parameter(graph, 1,
				&coordinate_buffer.buffer) == 0);
		assert(spa_fgn_graph_process(graph, centroid_inputs, 2,
				centroid_outputs, 3) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(offset_values[0] == 1.75f && offset_values[1] == 4.5f);
		assert(flux_value[0] == 4.0f && validity_value[0] == 1);

		pixel_values[0] = 1.0f;
		pixel_values[1] = 1.0f;
		pixel_values[2] = 1.0f;
		assert(spa_fgn_graph_process(graph, centroid_inputs, 2,
				centroid_outputs, 3) == 0);
		assert(offset_values[0] == 0.0f && offset_values[1] == 0.0f);
		assert(flux_value[0] == 3.0f && validity_value[0] == 0);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	const char *calibration_profiles = "";
	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = calibrate plugin = \"%s\""
		"   label = pixel-calibration-u16-f32"
		" config = { image_size = [ 1 3 ] "
		"              initial_flat = [ 2.0 1.0 0.5 ]"
		"              initial_background = [ 1.0 2.0 3.0 ]%s"
		"              rate = [ 1000 1 ] } }"
		"] inputs = [ \"calibrate:raw\" \"calibrate:flat\""
		"             \"calibrate:background\" ]"
		" outputs = [ \"calibrate:calibrated\" ] }",
		argv[1], calibration_profiles);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "pixel-calibration declaration graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 3);
	assert(spa_fgn_graph_get_n_outputs(graph) == 1);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_INPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_U16_LE &&
			format->n_dimensions == 2 && format->shape[0] == 1 &&
			format->shape[1] == 3);
	assert(strcmp(format->schema,
			"pipewireao/raw-detector-pixels-u16/1") == 0);
	assert(spa_fgn_graph_get_port_format(graph, SPA_DIRECTION_OUTPUT,
			0, &format) == 0);
	assert(format->element_type == SPA_ELEMENT_TYPE_F32_LE);

	{
		uint16_t raw_values[3] = { 4, 2, 1 };
		float flat_values[3] = { 1.0f, 2.0f, 4.0f };
		float background_values[3] = { 0.0f, 0.0f, 0.0f };
		float invalid_background_values[3] = { 0.0f, NAN, 0.0f };
		float calibrated_values[3] = { 0.0f };
		struct raw_test_buffer raw_buffer, flat_buffer, background_buffer;
		struct raw_test_buffer invalid_background_buffer;
		struct raw_test_buffer calibrated_buffer;
		struct spa_buffer *calibration_inputs[3];
		struct spa_buffer *calibration_outputs[1];
		struct spa_fgn_parameter_update updates[2];
		struct spa_fgn_parameter_update invalid_updates[2];

		init_raw_buffer(&raw_buffer, raw_values, sizeof(raw_values));
		init_raw_buffer(&flat_buffer, flat_values, sizeof(flat_values));
		init_raw_buffer(&background_buffer, background_values,
				sizeof(background_values));
		init_raw_buffer(&invalid_background_buffer, invalid_background_values,
				sizeof(invalid_background_values));
		init_raw_buffer(&calibrated_buffer, calibrated_values,
				sizeof(calibrated_values));
		raw_buffer.header.seq = 111;
		calibration_inputs[0] = &raw_buffer.buffer;
		calibration_inputs[1] = NULL;
		calibration_inputs[2] = NULL;
		calibration_outputs[0] = &calibrated_buffer.buffer;

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, calibration_inputs, 3,
				calibration_outputs, 1) == 0);
		assert(calibrated_values[0] == 6.0f &&
				calibrated_values[1] == 0.0f &&
				calibrated_values[2] == -1.0f);
		assert(calibrated_buffer.header.seq == 111);

		updates[0] = (struct spa_fgn_parameter_update) {
			.input_port = 1,
			.buffer = &flat_buffer.buffer,
		};
		updates[1] = (struct spa_fgn_parameter_update) {
			.input_port = 2,
			.buffer = &background_buffer.buffer,
		};
		assert(spa_fgn_graph_set_parameters(graph, updates, 2) == 0);
		assert(spa_fgn_graph_set_parameters(graph, updates, 2) == -EBUSY);
		assert(spa_fgn_graph_process(graph, calibration_inputs, 3,
				calibration_outputs, 1) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(calibrated_values[0] == 4.0f &&
				calibrated_values[1] == 4.0f &&
				calibrated_values[2] == 4.0f);

		invalid_updates[0] = updates[0];
		invalid_updates[1] = (struct spa_fgn_parameter_update) {
			.input_port = 2,
			.buffer = &invalid_background_buffer.buffer,
		};
		assert(spa_fgn_graph_set_parameters(graph, invalid_updates, 2) == -EINVAL);
		assert(spa_fgn_graph_process(graph, calibration_inputs, 3,
				calibration_outputs, 1) == 0);
		assert(calibrated_values[0] == 4.0f &&
				calibrated_values[1] == 4.0f &&
				calibrated_values[2] == 4.0f);

		flat_buffer.header.seq = 201;
		background_buffer.header.seq = 202;
		assert(spa_fgn_graph_set_parameters(graph, updates, 2) == -EINVAL);
		assert(spa_fgn_graph_process(graph, calibration_inputs, 3,
				calibration_outputs, 1) == 0);
		assert(calibrated_values[0] == 4.0f &&
				calibrated_values[1] == 4.0f &&
				calibrated_values[2] == 4.0f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = first-correction plugin = \"%s\""
		"   label = optical-gain-correction-f32"
		" config = { optical_gains = [ 2.0 ] rate = [ 1000 1 ] } }"
		" { type = ndarray name = second-correction plugin = \"%s\""
		"   label = optical-gain-correction-f32"
		" config = { optical_gains = [ 3.0 ] rate = [ 1000 1 ] } }"
		"] links = ["
		" { output = \"first-correction:corrected\""
		"   input = \"second-correction:reconstructed\" }"
		"] inputs = [ \"first-correction:reconstructed\""
		"             \"first-correction:optical-gains\""
		"             \"second-correction:optical-gains\" ]"
		" outputs = [ \"second-correction:corrected\" ] }",
		argv[1], argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	res = spa_fgn_graph_new(config, &graph);
	if (res < 0)
		fprintf(stderr, "cross-node parameter graph failed: %s (%d)\n",
				strerror(-res), res);
	assert(res == 0);
	assert(spa_fgn_graph_get_n_inputs(graph) == 3);
	assert(spa_fgn_graph_get_n_outputs(graph) == 1);

	{
		float reconstructed_value[1] = { 210.0f };
		float first_gain_value[1] = { 5.0f };
		float second_gain_value[1] = { 7.0f };
		float invalid_gain_value[1] = { 0.0f };
		float corrected_value[1] = { 0.0f };
		struct raw_test_buffer reconstructed_buffer, first_gain_buffer;
		struct raw_test_buffer second_gain_buffer, invalid_gain_buffer;
		struct raw_test_buffer corrected_buffer;
		struct spa_buffer *correction_inputs[3];
		struct spa_buffer *correction_outputs[1];
		struct spa_fgn_parameter_update updates[2];

		init_raw_buffer(&reconstructed_buffer, reconstructed_value,
				sizeof(reconstructed_value));
		init_raw_buffer(&first_gain_buffer, first_gain_value,
				sizeof(first_gain_value));
		init_raw_buffer(&second_gain_buffer, second_gain_value,
				sizeof(second_gain_value));
		init_raw_buffer(&invalid_gain_buffer, invalid_gain_value,
				sizeof(invalid_gain_value));
		init_raw_buffer(&corrected_buffer, corrected_value,
				sizeof(corrected_value));
		first_gain_buffer.header.seq = 301;
		second_gain_buffer.header.seq = 301;
		invalid_gain_buffer.header.seq = 302;
		correction_inputs[0] = &reconstructed_buffer.buffer;
		correction_inputs[1] = NULL;
		correction_inputs[2] = NULL;
		correction_outputs[0] = &corrected_buffer.buffer;
		updates[0] = (struct spa_fgn_parameter_update) {
			.input_port = 1,
			.buffer = &first_gain_buffer.buffer,
		};
		updates[1] = (struct spa_fgn_parameter_update) {
			.input_port = 2,
			.buffer = &second_gain_buffer.buffer,
		};

		assert(spa_fgn_graph_activate(graph) == 0);
		assert(spa_fgn_graph_process(graph, correction_inputs, 3,
				correction_outputs, 1) == 0);
		assert(corrected_value[0] == 35.0f);

		assert(spa_fgn_graph_set_parameters(graph, updates, 2) == 0);
		assert(spa_fgn_graph_process(graph, correction_inputs, 3,
				correction_outputs, 1) ==
				SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
		assert(fabsf(corrected_value[0] - 6.0f) < 1.0e-5f);

		updates[1].buffer = &invalid_gain_buffer.buffer;
		assert(spa_fgn_graph_set_parameters(graph, updates, 2) == -EINVAL);
		assert(spa_fgn_graph_process(graph, correction_inputs, 3,
				correction_outputs, 1) == 0);
		assert(fabsf(corrected_value[0] - 6.0f) < 1.0e-5f);
		assert(spa_fgn_graph_deactivate(graph) == 0);
	}
	spa_fgn_graph_free(graph);
	graph = NULL;

	res = snprintf(config, sizeof(config),
		"{ nodes = ["
		" { type = ndarray name = fast-loop plugin = \"%s\""
		"   label = leaky-integrator-f32"
		" config = { extent = 1 initial_state = 0.0"
		"              input_schema = pipewireao/controller-residual-error-f32/1"
		"              output_schema = pipewireao/controller-command-f32/1"
		"              rate = [ 1000 1 ] } }"
		" { type = ndarray name = slow-loop plugin = \"%s\""
		"   label = leaky-integrator-f32"
		" config = { extent = 1 initial_state = 0.0"
		"              input_schema = pipewireao/controller-residual-error-f32/1"
		"              output_schema = pipewireao/controller-command-f32/1"
		"              rate = [ 500 1 ] } }"
		"] inputs = [ \"fast-loop:input\" \"slow-loop:input\" ]"
		" outputs = [ \"fast-loop:output\" \"slow-loop:output\" ] }",
		argv[1], argv[1]);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == -EINVAL);
	assert(graph == NULL);
	return 0;
}
