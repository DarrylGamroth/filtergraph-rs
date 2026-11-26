/* SPDX-License-Identifier: MIT */
/* Current JFG physical-reference snapshot and finite-only validity contracts. */
static void test_scientific_contract_declarations(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[2200];
	float residual[3] = { 2, -3, 0.25f }, compensated[3] = { 71, 71, 71 };
	float physical[3] = { 73, 73, 73 }, offset[3] = { 1, -2, 0.5f };
	float q[6] = { 1, 2, -1, 0.5f, -2, 3 }, c[6] = { 1, -0.5f, 2, 1, -1, 3 };
	struct raw_test_buffer input, output, physical_output, offset_buffer, q_buffer, c_buffer;
	struct spa_buffer *inputs[] = { &input.buffer, NULL, NULL, NULL };
	struct spa_buffer *outputs[] = { &output.buffer, &physical_output.buffer };
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = reference plugin = \"%s\" label = physical-offset-reference-f32"
		" config = { residual_count = 3 physical_count = 3 mode_count = 2 reference_sign = -1"
		" residual_schema = org.example.residual/1 physical_schema = org.example.physical/1"
		" physical_to_modes_schema = org.example.q/1 modes_to_reference_schema = org.example.c/1 } } ]"
		" inputs = [ \"reference:reconstructed-residual\" \"reference:offset\" \"reference:physical-to-modes\" \"reference:modes-to-reference\" ]"
		" outputs = [ \"reference:compensated-residual\" \"reference:physical-offset\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config));
	assert(spa_fgn_graph_new(config, &graph) == 0);
	init_raw_buffer(&input, residual, sizeof(residual)); init_raw_buffer(&output, compensated, sizeof(compensated));
	init_raw_buffer(&physical_output, physical, sizeof(physical)); init_raw_buffer(&offset_buffer, offset, sizeof(offset));
	init_raw_buffer(&q_buffer, q, sizeof(q)); init_raw_buffer(&c_buffer, c, sizeof(c)); input.header.seq = 901;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 4, outputs, 2) == 0);
	assert(compensated[0] == 2 && physical[0] == 0);
	struct spa_fgn_parameter_update updates[] = {
		{ .input_port = 2, .buffer = &q_buffer.buffer },
		{ .input_port = 3, .buffer = &c_buffer.buffer },
		{ .input_port = 1, .buffer = &offset_buffer.buffer },
	};
	assert(spa_fgn_graph_set_parameters(graph, updates, 3) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 4, outputs, 2) == SPA_FGN_PROCESS_RESULT_PROPS_CHANGED);
	assert(compensated[0] == 8.5f && compensated[1] == -2 && compensated[2] == -21.25f);
	assert(physical[0] == 1 && physical[1] == -2 && physical[2] == 0.5f);
	assert(output.header.seq == 901 && physical_output.header.seq == 901);
	offset[0] = NAN; assert(spa_fgn_graph_update_parameter(graph, 1, &offset_buffer.buffer) < 0);
	residual[0] = NAN; input.header.seq = 902;
	assert(spa_fgn_graph_process(graph, inputs, 4, outputs, 2) < 0);
	assert(compensated[0] == 8.5f && physical[0] == 1 && output.header.seq == 0 && physical_output.header.seq == 0);
	residual[0] = 2; assert(spa_fgn_graph_reset(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 4, outputs, 2) == 0);
	assert(compensated[0] == 8.5f && physical[0] == 1 && output.header.seq == 902);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

	res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = finite plugin = \"%s\" label = pixel-validity-f32"
		" config = { extent = 6 input_schema = org.example.adu/1 } } ]"
		" inputs = [ \"finite:input\" ] outputs = [ \"finite:validity\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	float pixels[6] = { 0, -0.0f, -3.5f, NAN, INFINITY, -INFINITY };
	uint8_t validity[6] = { 0 };
	init_raw_buffer(&input, pixels, sizeof(pixels)); init_raw_buffer(&output, validity, sizeof(validity));
	input.header.seq = 911;
	assert(spa_fgn_graph_activate(graph) == 0);
	assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
	assert(validity[0] && validity[1] && validity[2] && !validity[3] && !validity[4] && !validity[5]);
	assert(output.header.seq == 911);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}

static void test_acquisition_chopped_declaration(const char *plugin)
{
	struct spa_fgn_graph *graph = NULL;
	char config[1200];
	int res = snprintf(config, sizeof(config),
		"{ nodes = [ { type = ndarray name = chopped plugin = \"%s\" label = chopped-image-difference-f32"
		" config = { pixel_count = 2 trigger_high_is_fringed = true input_schema = org.example.adu/1 require_acquisition_identity = true } } ]"
		" inputs = [ \"chopped:image\" \"chopped:input-validity\" ]"
		" outputs = [ \"chopped:difference\" \"chopped:output-validity\" \"chopped:pair-status\" ] }", plugin);
	assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
	float image[2] = { 10, 20 }, difference[2] = { 71, 71 };
	uint8_t valid[2] = { 1, 1 }, output_validity[2] = { 1, 1 }, status[1] = { 99 };
	struct spa_meta_acquisition_identity acquisition = { .flags = SPA_META_RECORD_VALID,
        .domain = { 1 }, .generation = 1 };
    struct spa_meta_exposure_timing exposure = { .flags = SPA_META_EXPOSURE_DURATION_VALID,
        .duration_nsec = 20 };
    struct spa_meta_optical_chopping optical = { .flags = SPA_META_RECORD_VALID };
    struct spa_meta metas[4];
	struct raw_test_buffer input, validity_input, output, validity_output, status_output;
	struct spa_buffer *inputs[] = { &input.buffer, &validity_input.buffer };
	struct spa_buffer *outputs[] = { &output.buffer, &validity_output.buffer, &status_output.buffer };
	init_raw_buffer(&input, image, sizeof(image)); init_raw_buffer(&validity_input, valid, sizeof(valid));
	init_raw_buffer(&output, difference, sizeof(difference)); init_raw_buffer(&validity_output, output_validity, sizeof(output_validity));
	init_raw_buffer(&status_output, status, sizeof(status));
	metas[0] = input.meta;
    metas[1] = (struct spa_meta){ SPA_META_AcquisitionIdentity, sizeof(acquisition), &acquisition };
    metas[2] = (struct spa_meta){ SPA_META_ExposureTiming, sizeof(exposure), &exposure };
    metas[3] = (struct spa_meta){ SPA_META_OpticalChopping, sizeof(optical), &optical };
    input.buffer.metas = metas; input.buffer.n_metas = 4;
    struct spa_meta output_metas[3][4];
    struct spa_meta_acquisition_identity output_identity[3] = { 0 };
    struct spa_meta_exposure_timing output_exposure[3] = { 0 };
    struct spa_meta_optical_chopping output_optical[3] = { 0 };
    for (size_t i = 0; i < 3; i++) {
        output_metas[i][0] = outputs[i]->metas[0];
        output_metas[i][1] = (struct spa_meta){ SPA_META_AcquisitionIdentity, sizeof(output_identity[i]), &output_identity[i] };
        output_metas[i][2] = (struct spa_meta){ SPA_META_ExposureTiming, sizeof(output_exposure[i]), &output_exposure[i] };
        output_metas[i][3] = (struct spa_meta){ SPA_META_OpticalChopping, sizeof(output_optical[i]), &output_optical[i] };
        outputs[i]->metas = output_metas[i]; outputs[i]->n_metas = 4;
    }
    assert(spa_fgn_graph_activate(graph) == 0);
	const uint64_t sequences[] = { 1, 2, 4, 5, 6, 7, 8 };
	const uint32_t kinds[] = { SPA_META_OPTICAL_FRINGED, SPA_META_OPTICAL_UNFRINGED,
        SPA_META_OPTICAL_FRINGED, SPA_META_OPTICAL_UNFRINGED, SPA_META_OPTICAL_MIXED,
        SPA_META_OPTICAL_FRINGED, SPA_META_OPTICAL_UNFRINGED };
	const uint8_t statuses[] = { 0, 1, 0, 1, 0, 0, 1 };
	for (size_t i = 0; i < sizeof(sequences)/sizeof(sequences[0]); i++) {
		input.header.seq = sequences[i]; image[0] = (float)sequences[i];
		acquisition.sequence = sequences[i];
        optical.classification = kinds[i];
		assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 3) == 0);
		assert(status[0] == statuses[i] && output_validity[0] == statuses[i]);
		assert(output.header.seq == sequences[i]);
	}
	float retained_difference = difference[0];
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 3) < 0);
	assert(difference[0] == retained_difference && status[0] == 1 && output.header.seq == 0);
	assert(spa_fgn_graph_reset(graph) == 0);
	input.header.seq = 1; acquisition.generation = 2; acquisition.sequence = 1;
	assert(spa_fgn_graph_process(graph, inputs, 2, outputs, 3) == 0 && status[0] == 0);
	assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}
