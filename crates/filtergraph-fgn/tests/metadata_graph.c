/* SPDX-License-Identifier: MIT */
/* Native 4.x ABI admission and facts through two internal Rust copy nodes. */
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <spa/filter-graph/filter-graph-ndarray.h>

_Static_assert(SPA_FGN_PLUGIN_ABI_VERSION == 8, "current FGN ABI");
_Static_assert(sizeof(struct spa_fgn_format) == 40, "format ABI");
_Static_assert(sizeof(struct spa_fgn_port_info) == 40, "port ABI");
_Static_assert(offsetof(struct spa_fgn_port_info, n_metas) == 24, "port metadata count");
_Static_assert(offsetof(struct spa_fgn_port_info, propagation) == 28, "port propagation");
_Static_assert(offsetof(struct spa_fgn_port_info, metas) == 32, "port requirements");

struct frame {
    struct spa_buffer buffer;
    struct spa_data data;
    struct spa_chunk chunk;
    struct spa_meta metas[7];
    struct spa_meta_header header;
    struct spa_meta_acquisition_identity identity;
    struct spa_meta_exposure_timing exposure;
    struct spa_meta_reference_timestamp reference;
    struct spa_meta_optical_chopping optical;
    struct spa_meta_electrical_trigger trigger;
    uint64_t opaque;
    float values[2];
};
static void init(struct frame *f)
{
    memset(f, 0, sizeof(*f));
    f->chunk = (struct spa_chunk){ .size = sizeof(f->values), .stride = sizeof(float) };
    f->data = (struct spa_data){ .type = SPA_DATA_MemPtr, .flags = SPA_DATA_FLAG_READWRITE,
        .fd = -1, .maxsize = sizeof(f->values), .data = f->values, .chunk = &f->chunk };
    f->metas[0] = (struct spa_meta){ SPA_META_Header, sizeof(f->header), &f->header };
    f->metas[1] = (struct spa_meta){ SPA_META_AcquisitionIdentity, sizeof(f->identity), &f->identity };
    f->metas[2] = (struct spa_meta){ SPA_META_ExposureTiming, sizeof(f->exposure), &f->exposure };
    f->metas[3] = (struct spa_meta){ SPA_META_ReferenceTimestamp, sizeof(f->reference), &f->reference };
    f->metas[4] = (struct spa_meta){ SPA_META_OpticalChopping, sizeof(f->optical), &f->optical };
    f->metas[5] = (struct spa_meta){ SPA_META_ElectricalTrigger, sizeof(f->trigger), &f->trigger };
    f->metas[6] = (struct spa_meta){ 0x777, sizeof(f->opaque), &f->opaque };
    f->buffer = (struct spa_buffer){ .n_metas = 7, .metas = f->metas, .n_datas = 1, .datas = &f->data };
}
int main(int argc, char **argv)
{
    assert(argc == 2);
    char config[4096];
    int n = snprintf(config, sizeof(config),
        "{ nodes = [ { type = ndarray name = a plugin = \"%s\" label = ndarray-copy-f32"
        " config = { extent = 2 schema = org.example.metadata/1 } }"
        " { type = ndarray name = b plugin = \"%s\" label = ndarray-copy-f32"
        " config = { extent = 2 schema = org.example.metadata/1 } } ]"
        " links = [ { output = \"a:output\" input = \"b:input\" } ]"
        " inputs = [ \"a:input\" ] outputs = [ \"b:output\" ] }", argv[1], argv[1]);
    assert(n > 0 && (size_t)n < sizeof(config));
    struct spa_fgn_graph *graph = NULL;
    int res = spa_fgn_graph_new(config, &graph);
    if (res < 0) fprintf(stderr, "metadata graph: %s (%d)\n", strerror(-res), res);
    assert(res == 0);
    struct frame input, output;
    init(&input); init(&output);
    input.header = (struct spa_meta_header){ .pts = 123, .seq = 901 };
    input.identity = (struct spa_meta_acquisition_identity){ .flags = SPA_META_RECORD_VALID,
        .domain = {1}, .generation = 4, .sequence = 19 };
    input.exposure = (struct spa_meta_exposure_timing){ .flags = SPA_META_EXPOSURE_START_VALID | SPA_META_EXPOSURE_DURATION_VALID,
        .clock = SPA_META_CLOCK_MODEL, .start_nsec = -100, .duration_nsec = 17 };
    input.reference = (struct spa_meta_reference_timestamp){ .clock = SPA_META_CLOCK_TAI,
        .timestamp_nsec = UINT64_C(18000000000000001123), .duration_nsec = UINT64_MAX };
    input.optical = (struct spa_meta_optical_chopping){ .flags = SPA_META_RECORD_VALID, .classification = SPA_META_OPTICAL_FRINGED };
    input.trigger = (struct spa_meta_electrical_trigger){ .flags = SPA_META_RECORD_VALID, .classification = SPA_META_TRIGGER_LOW };
    input.opaque = UINT64_C(0x123456789abcdef0); input.values[0] = 3; input.values[1] = -7;
    struct spa_buffer *inputs[] = { &input.buffer }, *outputs[] = { &output.buffer };
    struct spa_meta descriptors[7]; memcpy(descriptors, output.metas, sizeof(descriptors));
    assert(spa_fgn_graph_activate(graph) == 0);
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
    assert(output.values[0] == 3 && output.values[1] == -7);
    assert(output.header.seq == 901 && output.identity.sequence == 19);
    assert(output.header.pts == 123 && output.exposure.start_nsec == -100);
    assert(output.reference.timestamp_nsec == input.reference.timestamp_nsec && output.reference.clock == SPA_META_CLOCK_TAI);
    assert(output.optical.classification == SPA_META_OPTICAL_FRINGED && output.trigger.classification == SPA_META_TRIGGER_LOW);
    assert(output.opaque == input.opaque && memcmp(output.metas, descriptors, sizeof(descriptors)) == 0);
    /* Omitted optional records clear preallocated output slots on reuse. */
    input.buffer.n_metas = 2;
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) == 0);
    assert(output.header.seq == 901 && output.identity.sequence == 19);
    assert(output.exposure.flags == 0 && output.reference.clock == 0);
    assert(output.optical.flags == 0 && output.trigger.flags == 0 && output.opaque == 0);
    assert(memcmp(output.metas, descriptors, sizeof(descriptors)) == 0);
    /* A negotiated required identity must contain a fact. */
    input.identity.flags = 0; memset(&input.identity, 0, sizeof(input.identity));
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
    input.buffer.n_metas = 1;
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
    input.buffer.n_metas = 2; input.identity.flags = SPA_META_RECORD_VALID; input.identity.domain[0] = 1;
    input.metas[1].type = 11;
    assert(spa_fgn_graph_process(graph, inputs, 1, outputs, 1) < 0);
    assert(spa_fgn_graph_deactivate(graph) == 0);
    spa_fgn_graph_free(graph);
    return 0;
}
