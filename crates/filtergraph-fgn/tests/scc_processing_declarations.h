/* SPDX-License-Identifier: MIT */
/* SCC coordinates, bounded solves, rejection publication, and metadata recovery. */
static void test_scc_processing_declarations(const char *plugin)
{
 struct spa_fgn_graph *graph = NULL;
 char config[2800];
 float difference[6] = { 1, 2, 4, 8, 16, 32 }, measurements[12], weights[12], fraction = -1;
 bool validity[6] = { true, true, true, true, true, true }, measurement_validity[12];
 uint8_t pair = 1;
 struct raw_test_buffer difference_buffer, validity_buffer, pair_buffer, measurement_buffer, weight_buffer, measurement_validity_buffer, fraction_buffer;
 struct spa_buffer *inputs[] = { &difference_buffer.buffer, &validity_buffer.buffer, &pair_buffer.buffer };
 struct spa_buffer *outputs[] = { &measurement_buffer.buffer, &weight_buffer.buffer, &measurement_validity_buffer.buffer, &fraction_buffer.buffer };
 int res = snprintf(config, sizeof(config),
  "{ nodes = [ { type = ndarray name = s plugin = \"%s\" label = scc-spatial-carrier-demodulation-f32"
  " config = { image_size = [ 2 3 ] roi = [ true true true true true true ] fir_tap_count = 1"
  " initial_fir = [ 1 ] minimum_support = 1 minimum_valid_fraction = 1  initial_reference_difference = null  initial_static_validity = null  initial_carrier_phase = null } } ]"
  " inputs = [ \"s:difference\" \"s:difference-validity\" \"s:pair-status\" ]"
  " outputs = [ \"s:measurements\" \"s:weights\" \"s:measurement-validity\" \"s:valid-fraction\" ] }", plugin);
 assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
 init_raw_buffer(&difference_buffer, difference, sizeof(difference)); init_raw_buffer(&validity_buffer, validity, sizeof(validity)); init_raw_buffer(&pair_buffer, &pair, sizeof(pair));
 init_raw_buffer(&measurement_buffer, measurements, sizeof(measurements)); init_raw_buffer(&weight_buffer, weights, sizeof(weights)); init_raw_buffer(&measurement_validity_buffer, measurement_validity, sizeof(measurement_validity)); init_raw_buffer(&fraction_buffer, &fraction, sizeof(fraction));
 difference_buffer.header.seq = 231;
 assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 4) == 0);
 for (size_t i = 0; i < 12; ++i) { assert(measurements[i] == (i < 6 ? difference[i] : 0)); assert(weights[i] == 1 && measurement_validity[i]); }
 assert(fraction == 1 && measurement_buffer.header.seq == 231 && weight_buffer.header.seq == 231 && measurement_validity_buffer.header.seq == 231 && fraction_buffer.header.seq == 231);
 pair = 0; difference[0] = 3; difference_buffer.header.seq = 232;
 assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 4) < 0); assert(measurements[0] == 1 && fraction == 1 && measurement_buffer.header.seq == 0);
 pair = 1; validity[0] = false;
 assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 4) < 0); assert(measurements[0] == 1 && measurement_buffer.header.seq == 0);
 validity[0] = true; assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 4) == 0); assert(measurements[0] == 3 && measurement_buffer.header.seq == 232);
 assert(spa_fgn_graph_reset(graph) == 0); assert(spa_fgn_graph_process(graph, inputs, 3, outputs, 4) == 0);
 assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

 float samples[2] = { 2, -3 }, dynamic_weights[2] = { 1, 2 }, command[2] = { 71, 71 };
 struct raw_test_buffer samples_buffer, dynamic_weights_buffer, command_buffer;
 struct spa_buffer *pcg_inputs[] = { &samples_buffer.buffer, &dynamic_weights_buffer.buffer };
 struct spa_buffer *pcg_outputs[] = { &command_buffer.buffer };
 res = snprintf(config, sizeof(config),
  "{ nodes = [ { type = ndarray name = p plugin = \"%s\" label = scc-weighted-regularized-reconstructor-f32"
  " config = { measurement_count = 2 controller_coordinate_count = 2 regularization_row_count = 2 measurement_schema = org.example.scc/1"
  " initial_interaction_matrix = [ 1 0 0 1 ] initial_regularization_operator = [ 1 0 0 1 ] regularization = 0.25 iterations = 8 relative_tolerance = 0.00001 minimum_diagonal = 0.00000001 } } ]"
  " inputs = [ \"p:measurements\" \"p:weights\" ] outputs = [ \"p:controller-residual-error\" ] }", plugin);
 assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
 init_raw_buffer(&samples_buffer, samples, sizeof(samples)); init_raw_buffer(&dynamic_weights_buffer, dynamic_weights, sizeof(dynamic_weights)); init_raw_buffer(&command_buffer, command, sizeof(command));
 samples_buffer.header.seq = 241; assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, pcg_inputs, 2, pcg_outputs, 1) == 0);
 assert(fabsf(command[0] - 1.6f) < 0.00001f && fabsf(command[1] + 8.0f/3) < 0.00001f && command_buffer.header.seq == 241);
 dynamic_weights[1] = -1; samples_buffer.header.seq = 242; assert(spa_fgn_graph_process(graph, pcg_inputs, 2, pcg_outputs, 1) < 0); assert(command_buffer.header.seq == 0 && fabsf(command[0] - 1.6f) < 0.00001f);
 dynamic_weights[1] = 2; assert(spa_fgn_graph_process(graph, pcg_inputs, 2, pcg_outputs, 1) == 0); assert(command_buffer.header.seq == 242);
 assert(spa_fgn_graph_reset(graph) == 0); assert(spa_fgn_graph_process(graph, pcg_inputs, 2, pcg_outputs, 1) == 0);
 assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);

 /* Optical role remains application-local; a bare SPA header cannot authorize SCC correction. */
 float frame[4] = { 1, 2, 3, 4 };
 struct raw_test_buffer frame_buffer;
 struct spa_buffer *correction_inputs[] = { &frame_buffer.buffer };
 res = snprintf(config, sizeof(config),
  "{ nodes = [ { type = ndarray name = c plugin = \"%s\" label = scc-correction-f32 config = { frame_size = [ 2 2 ] controller_coordinate_count = 2 image_schema = org.example.frame/1"
  " exposure_duration_nanoseconds = 1000 coordinate_order = [ 1 2 ] command_unit = controller-unit reconstructor_sign = command_error"
  " reference_difference = [ 0 0 0 0 ] correction_mask = [ true false false true ] reconstructor = [ 1 0 0 1 ] fringed_response = [ 0 0 0 0 ] unfringed_response = [ 0 0 0 0 ]"
  " bias_command = [ 0.25 -0.5 ] command_limits = [ -10 10 ] gain = 0.125 pole = 0.75 } } ] inputs = [ \"c:frame\" ] outputs = [ \"c:command\" ] }", plugin);
 assert(res > 0 && (size_t)res < sizeof(config)); assert(spa_fgn_graph_new(config, &graph) == 0);
 command[0] = command[1] = 71; init_raw_buffer(&frame_buffer, frame, sizeof(frame)); init_raw_buffer(&command_buffer, command, sizeof(command));
 frame_buffer.header.seq = 251; command_buffer.header.seq = 99;
 assert(spa_fgn_graph_activate(graph) == 0); assert(spa_fgn_graph_process(graph, correction_inputs, 1, pcg_outputs, 1) < 0); assert(command[0] == 71 && command[1] == 71 && command_buffer.header.seq == 0);
 assert(spa_fgn_graph_reset(graph) == 0); assert(spa_fgn_graph_process(graph, correction_inputs, 1, pcg_outputs, 1) < 0);
 assert(spa_fgn_graph_deactivate(graph) == 0); spa_fgn_graph_free(graph);
}
