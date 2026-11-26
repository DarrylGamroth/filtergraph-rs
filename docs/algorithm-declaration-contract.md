# FilterGraph algorithm declaration contract

Status: active proof contract

Review date: 2026-08-27


Scope update (2026-10-25): this repository delivers Rust implementations only.
References to Julia describe comparison with the external JFG implementation,
not an in-repository package or runtime dependency. Stable requirement IDs and
scientific contracts are retained.

## Authority and scope

This document defines the scientist-facing boundary shared by native FilterGraph
execution and deployment adapters. Uppercase requirement terms use the
meanings defined by BCP 14 (RFC 2119 and RFC 8174).

The scientific implementation is an ordinary typed operation with an
immutable prepared plan, exclusively owned mutable workspace, borrowed inputs,
and caller-owned outputs. It MAY implement `AlgorithmPlan`, but the graph
declaration MUST NOT require an otherwise suitable operation to adopt that
trait. The scalar-property contract remains authoritative for scalar
descriptors and updates. The FGN contract remains authoritative for its C ABI,
graph host, and real-time callback behavior.

The declaration is scientific metadata and typed glue, not a plugin API. Its
vocabulary is limited to configuration, ports, logical shapes, scientific
schemas, properties, sparse parameters, sample identity, preparation,
processing, publication outcome, and reset. PipeWire rate is a deployment
field applied by the adapter; a declaration can state an exact rational rate
ratio between a port and that deployment cadence without naming PipeWire.
An optional operation-shape marker names a mathematical primitive such as
row-major dense F32 multiplication; it does not select a worker count,
scheduler, transport, or thread policy.

## Requirements

### CAL-DECL-001 — Scientist-sized authoring surface

For each language implementation, an algorithm prepared for graph use MUST
declare its canonical label, typed construction configuration, ports, logical
shapes, scientific schemas, scalar-property policy, and sparse parameter
replacement at most once, beside its scientific implementation. The author
MUST write only ordinary typed plan/workspace preparation and processing code
plus this declaration. The author MUST NOT write a deployment registry entry,
transport-specific decoder, ABI descriptor, raw-pointer or buffer conversion,
C callback, errno mapping, unwind barrier, publication state machine, or
real-time memory-management code.

A deployment maintainer MAY select declaration types for a particular shared
library. That selection MUST contain types or labels only and MUST NOT repeat
algorithm facts or adaptation logic.

Verification intent: inspect complete Rust and Julia examples and the generated
shared-library bundle; identify every source edit needed to add one algorithm.

### CAL-DECL-002 — Explicit scientific roles and layout

The declaration MUST distinguish per-frame inputs and outputs, sparse ndarray
parameters, runtime scalar properties, construction configuration, and values
whose change requires graph reconstruction. Every ndarray port MUST declare a
logical element type, rank and shape, and versioned scientific schema. A fixed
operation MAY declare that schema directly. A generic operation whose input or
output meaning depends on its use MUST bind each independent schema through
explicit graph-rebuild configuration and MUST NOT substitute an algorithm
label or deployment name for the value's scientific meaning. A
deployment format MUST additionally expose its actual packed storage order.
Language-native storage MAY differ—ordinary Rust slices are row-major and
ordinary Julia matrices are column-major—without changing logical axes or
schema meaning. Shape arithmetic MUST reject zero, overflow, and values not
representable by the selected ABI before activation.

Verification intent: enumerate declarations, inspect multi-dimensional Rust
and Julia formats, prepare invalid dimensions, and compare native logical
results.

### CAL-DECL-003 — Native execution

The declaration MUST prepare, process, reset, and replace sparse parameters
without PipeWire, SPA, FGN, JSON, C callbacks, or dynamic-library loading.
Native callers MUST supply ordinary typed arrays or borrowed views and MUST
receive the same numerical result and state transition as a direct call to the
declared typed operation. An accepted sparse replacement MUST preserve every
prepared port format and remain compatible with the existing workspace.

Verification intent: execute stateful and parameterized declarations natively
and compare them with direct typed-operation calls over multiple samples.

### CAL-DECL-004 — Generated FGN adaptation

Shared infrastructure MUST derive the complete FGN descriptor and adapter from
the portable declaration. FGN adaptation MUST preserve construction, ports,
formats, properties, parameters, lifecycle, errors, metadata policy, and
process-boundary update adoption without algorithm-specific deployment code.
The standard ndarray rate deployment field MUST be handled once by shared
infrastructure and MUST NOT appear in scientific configuration types. Each data
input MUST project its declared required and optional sample facts by port name;
required layouts MUST be checked during preparation and required facts before
processing. Optional facts MUST retain their independent presence and value
semantics. A format MUST NOT carry
an interpretation-profile pointer. Deployment configuration MUST reject any
legacy `profiles` setting, including an empty object; it MUST NOT translate an
old `interpretation_profile` assignment. The adapter MUST preserve the declared
metadata source and propagation policy for each output.

The declaration surface MUST expose transport-independent queries for
configuration defaults and requiredness, preserving the distinction between a
required field without a default and a default value equal to `nothing`/`None`. Preparation
MUST expose actual per-instance metadata queries and immutable prepared layouts
needed by native consumers. These queries MUST use the same configuration
deserialization and metadata preparation behavior as execution.

Verification intent: load generated descriptors through the C graph host,
exercise construction, exact per-port rate/schema/metadata projection,
properties, parameters, lifecycle, and processing, and compare results with
native execution.

### CAL-DECL-005 — Rust and Julia logical equivalence

Rust and Julia declarations of the same algorithm MUST use the same canonical
label, configuration names and meanings, port names and roles, logical shapes,
scientific schemas, property descriptors, numerical behavior, and state
transitions. Catalog comparison includes configuration order, defaults and
requiredness, and prepared logical formats, metadata facts and Property
descriptors. Language macros MAY generate language-native types, dispatch, and
packed storage order provided these observable logical contracts are
equivalent and the actual deployment layout is explicit.

Verification intent: compare declaration metadata and shared stateful
numerical fixtures in Rust and Julia, including at least one multidimensional
case before a Julia FGN adapter is admitted.

### CAL-DECL-006 — Generated deployment inventory

A deployment build MUST derive descriptor and catalog facts from selected
declarations. Adding an algorithm to a deployment MAY require selecting its
declaration type at build time, but MUST NOT require repeating ports,
configuration, schemas, properties, shapes, or adaptation logic. A generated
library MUST export only the versioned FGN entry point required by the host.

Verification intent: compare the selected declaration set with the exported
descriptor registry, reject duplicate labels, and inspect dynamic symbols.

### CAL-DECL-007 — Admitted repeated path

Native and FGN processing through a prepared declaration MUST perform bounded
work without allocation, blocking, locks, system calls, destruction of
prepared state, or unwinding. FGN property and parameter adoption at the frame
boundary has the same constraint. Declaration parsing, shape validation,
preparation, parameter replacement, deployment selection, and reclamation
remain off that path. Serialized reset SHOULD reuse prepared storage.

Verification intent: run warmed allocation/deallocation tests around native
processing, reset for stateful declarations, and the complete generated FGN
callback; combine those tests with callback-containment, concurrency, and
system-call evidence rather than treating allocator counters as sufficient.

### CAL-DECL-008 — Scientific operation boundary

A declaration MUST be able to preserve one semantically atomic scientific
operation with multiple inputs, outputs, properties, sparse parameters, and
carried state. Migration to a graph MUST NOT force an operation into smaller
nodes when that split would expose a partial state or output commit, require an
atomic update mechanism the graph does not provide, or introduce feedback or
rate-transition semantics the graph does not provide. A decomposed form MUST
prove the same numerical, failure, and state-commit behavior as the atomic
typed operation before it replaces that operation.

Verification intent: declare closed-loop correction as one operation and
compare direct and declared execution across its first frame, delayed
constraint-feedback frames, state publication, reset, scalar replacement, and
sparse projection replacement. Treat the graph feedback delay and multi-rate
scheduling as separate host capabilities rather than silently approximating
them in the declaration.

### CAL-DECL-009 — Generic scientific-schema binding

A schema binding MUST be an ordinary nonempty versioned scientific-schema
identifier supplied in typed graph-rebuild configuration. A declaration MUST
associate the binding with its affected ports once and MUST preserve the exact
identifier in native and deployment formats. Rust and Julia implementations of
the same generic algorithm MUST use the same binding names and meanings. An
adapter MUST reject a missing, empty, malformed, or unknown configuration
field before activation. Changing a binding requires graph reconstruction and
MUST NOT be published as a runtime scalar-property or parameter update.

Verification intent: bind different input and output meanings for the generic
leaky integrator in Rust and Julia, bind the image-backed Shack–Hartmann input
and reconstructor output at instance construction, reject missing and empty
bindings, and prove that the C graph host rejects a link whose bound schemas
differ despite matching element type and shape.

### CAL-DECL-010 — Profile-defined generated graph placement

A deployment profile MUST be able to assemble generated declarations by label,
provide graph-rebuild configuration in the graph description, and provide
large prepared calibration values through typed parameter ports without
adding an algorithm-specific FGN adapter or editing the scientific algorithm.
All related parameter ports MUST be prepared as one validated startup
transaction while processing is inactive. The host MUST adopt that complete
transaction before invoking the first numerical callback.

A closed-loop profile MUST use the constraint feedback from the preceding
successful frame when its scientific plan requires delayed feedback.
Verification intent: load the production declaration bundle, prepare and
publish the typed startup parameters, and compare a synthetic graph with
the direct execution path at every declared boundary.

### CAL-DECL-011 — Conditional publication and sample identity

A declaration that consumes independently complete pieces of a larger
scientific sample MUST express that operation without transport-specific
metadata or partially published buffers. The portable declaration contract
MUST represent, at minimum, publication sequence, zero-based semantic offset,
optional timestamp, terminal indication, discontinuity, corruption, optional
physical acquisition identity, optional exposure start and duration, optional
reference timestamp, optical classification, and electrical trigger
classification. Physical acquisition sequence MUST be independent of
publication sequence. Trigger and optical classification presence MUST be
distinct from explicit `absent`, and each fact MUST retain its own presence.
Exposure clocks MUST distinguish monotonic, realtime, TAI, and model time;
start and duration MUST be independently optional. Native callers MUST be able
to supply the same compact facts that an FGN adapter obtains from standard
buffer metadata.

Processing MUST return an explicit complete or deferred publication outcome.
Every output that can be deferred MUST be declared conditional. A deferred
call MUST leave caller-owned output logically unpublished and MUST NOT cause a
required downstream operation to run. A completed aggregate MAY replace its
output sample metadata with a transport-neutral value derived from all pieces;
the deployment adapter MUST project that value onto standard output metadata.
Ordinary one-call/one-output declarations MUST continue to complete without
additional author code.

Every port whose cadence differs from the declaration's base deployment rate
MUST state an exact positive rational ratio. The adapter MUST reduce and
overflow-check the resulting rate. Rate conversion changes scheduling and
artifact granularity only; it MUST NOT change scientific array shapes or
schemas.

Verification intent: execute one declaration natively and through the C FGN
host over several immutable row blocks, verify no downstream call before the
terminal block, verify exact input and output rates, compare physical and
publication sequences, independent fact presence, propagated metadata and
discontinuity, and exercise malformed or missing metadata.

### CAL-DECL-012 — Row-block Shack–Hartmann reconstruction

The row-block Shack–Hartmann composition MUST consume complete calibrated
Float32 blocks with common shape `[N, detector_columns]`, where `N` is
positive, smaller than the detector height, and divides that height. It MUST
use one explicit zero-based `(row, column)` origin per uniform-size
subaperture. Declaration order defines slope and reconstructor-column order;
a prepared completion schedule MAY reorder readiness checks but MUST NOT
reorder scientific coordinates.

The composition MUST own bounded private frame, measurement, and
reconstruction workspace. It MUST measure each newly ready scientific prefix
and accumulate only the corresponding reconstructor columns. It MUST publish
one reconstructed frame result only after a contiguous, correctly identified,
terminal block completes the detector frame. Gaps, overlaps, changed sequence,
invalid terminal placement, mid-frame discontinuity, out-of-range blocks, and
corruption MUST abandon the partial frame without changing the published
output. The next completed output after a locally detected abandonment MUST be
marked discontinuous.

The composition MUST use separate sensing and incremental reconstruction
declarations. Sensing MUST publish a synchronized measurement, zero-based
column-index and count triplet for every accepted block, including count zero.
Its metadata offset MUST be the zero-based block ordinal, retaining the frame
sequence, timestamp, terminal and discontinuity fields. Reconstruction MUST
reject duplicate columns and publish only after complete column coverage and
the final expected block ordinal.

Subaperture origins, coordinates, reference slopes, thresholds and active mask
MUST remain sparse ndarray parameters of the sensing declaration; the
reconstructor MUST remain a sparse ndarray parameter of the reconstruction
declaration. Compatible replacements MUST rebuild a complete private plan
outside processing and preserve the prepared workspace and port formats.
Repeated block processing MUST meet CAL-DECL-007.

Verification intent: use explicit origins that skip detector rows and columns,
compare the terminal output with the full-frame image-view Shack–Hartmann plus
dense reconstruction path, test arbitrary declaration order and all recovery
conditions, and run allocator/deallocator checks after warmup.

### CAL-DECL-013 — Row-block pixel calibration

The row-block pixel-calibration declaration MUST consume one complete U16 ADU
block and publish one complete Float32 PDE block with common shape
`[N, detector_columns]`. Its input and output MUST use the established
`pipewireao/raw-pixel-row-block-u16/1` and
`pipewireao/calibrated-pixel-row-block-f32/1` schemas at the same block rate.
It MUST require transport-neutral sample identity, use the zero-based row
offset to select the corresponding detector-plane flat and background values,
and preserve sequence, offset, timestamp, terminal, corruption, and
discontinuity metadata.

Flat and background MUST remain complete detector-plane sparse parameters so
one calibration plan applies to every block. A replacement adopted after row
zero MUST take effect on that graph cycle and mark the first block using the
new plan discontinuous. A conforming frame aggregator MUST then abandon the
mixed partial frame. Native and FGN parameter replacement MUST invoke the same
plan-adoption hook. Repeated block processing and plan adoption MUST meet
CAL-DECL-007.

Verification intent: compare row windows with the complete-frame calibration
equation, reject invalid row quanta and sample metadata, exercise native and
FGN parameter adoption, link calibrated output directly to the row-block
Shack–Hartmann composition through the C graph host, and compare the terminal
result with the complete-frame path.

### CAL-DECL-014 — Row-block pyramid pupil-pixel normalization

The row-block pyramid composition MUST consume complete calibrated Float32
blocks with common shape `[N, detector_columns]` under the same fixed-height
and sample-identity rules as CAL-DECL-012. It MUST retain exactly four
explicit zero-based pupil origins, one common uniform pupil size, and one
shared row-major pupil mask. Pupil declaration order MUST define output order
and full-pupil accumulation order.

The composition MUST preserve HEART normalization semantics. Its first
completed frame MUST use zero normalization gain, each completed frame MUST
publish its current full-pupil mean, and the following frame MUST normalize
selected pupil pixels with the reciprocal mean committed by the preceding
successful frame. A dropped or numerically rejected frame MUST NOT replace
that committed gain.

The composition MUST own bounded private frame and output workspace. It MAY
process every complete contiguous row extent within any pupil before terminal
detector-frame completion and MUST NOT delay those coordinates merely because
an earlier pupil-major coordinate is not ready. It MUST process every selected
coordinate at most once, retain pupil-major output coordinates, and accumulate
the full-pupil mean in declaration order. It MUST publish the complete
pupil-major reconstruction-pixel array and
mean-pupil-intensity scalar together only after a contiguous terminal block
completes the frame. Both outputs MUST be conditional and MUST carry the same
completed sample identity.

Pupil origins and the pupil mask MUST remain sparse ndarray parameters. A
compatible plan replacement MUST preserve committed normalization history,
abandon any partial detector frame, and mark the next completed output
discontinuous. An explicit algorithm reset MUST also reset normalization
history to its zero-gain startup state. Repeated block processing MUST meet
CAL-DECL-007.

Verification intent: compare two consecutive row-processed frames with the
complete image-view pyramid operation, prove zero-gain startup and
previous-frame gain adoption, use pupil origins whose detector-arrival order
differs from pupil-major order, verify work from a partial pupil without early
publication, exercise mid-frame parameter adoption, check warmed
allocation/deallocation, and link row calibration to row pyramid processing
through the C graph host.

### CAL-DECL-015 — Optional native operation execution

A declaration MAY identify a supported mathematical operation shape once next
to its prepared plan. The marker MUST be transport-neutral, MUST be checked
against the exact plan type by the declaration machinery, and MUST NOT contain
a worker count, affinity, scheduling, NUMA, queue, callback, pointer, or ABI
concept. Omitting the marker MUST retain ordinary serial typed processing.

For `DenseF32`, the plan MUST be `PreparedGemv<f32>` and the declaration MUST
represent one F32 data input, one F32 data output, and the prepared row-major
matrix. Native FilterGraph processing MUST continue to use the declaration's
ordinary process body. An FGN adapter MAY route the marked operation to the
host executor, but MUST preserve input validation, output completion, metadata,
parameter adoption, numerical failure behavior, and the declared dense-F32
reduction profile. One complete task MUST be bitwise invariant across worker
counts on the same target. Splitting it into several accumulated column ranges
MAY change floating-point grouping and requires a separate progressive
numerical contract. The scientist MUST NOT implement an executor callback,
shard, barrier, atomic, thread, or FGN wrapper.

For `ShackHartmannImageF32`, the plan MUST be
`ShackHartmannImagePlan<f32>` and the declaration MUST represent one complete
F32 detector image, F32 pair-interleaved slopes, F32 subaperture fluxes, and
Bool8 subaperture validity. A native adapter MAY shard the declared
subaperture order because each measurement reads immutable image and plan
state and owns one disjoint output element in all three artifacts. It MUST
preserve the serial per-subaperture pixel traversal and reduction order, so a
complete result is bitwise invariant across worker counts on the same target.
Every multi-lane output boundary MUST isolate all three mutable artifacts on
cache-line boundaries; an adapter that cannot establish that ownership MUST
use the serial declaration. The ordinary typed declaration remains the
scientific reference and MUST NOT expose a lane range or worker policy.

A complete-frame pyramid wavefront-sensor reconstruction is `DenseF32` over
pupil-major `PYRAMID_RECONSTRUCTION_PIXELS_V1` input. Its declaration MUST
retain that sensor-specific input and reconstructor schema rather than relabel
the data as Shack-Hartmann slopes merely to reuse an adapter. The shared
mathematical marker, not a shared sensor-specific port declaration, supplies
native dense execution.

For `IncrementalDenseReconstructionF32`, the plan and workspace MUST be
`IncrementalDenseReconstructionPlan<f32>` and
`IncrementalDenseReconstructionWorkspace<f32>`. The declaration MUST represent
three synchronized inputs: F32 measurements, U32 zero-based reconstructor
column indices and a U32 count, plus one conditional reconstructed-vector
output. The typed state machine MUST retain sample admission, strict block
ordinal order, unique scientific-column coverage, private accumulation state,
finite checks, terminal publication and abandonment. A native adapter MAY
replace only accumulation of the accepted indexed measurement batch. Each lane
MUST own a disjoint reconstructed-output range with cache-line-isolated
internal boundaries and MUST preserve the serial batch order for every output.

For `PyramidRowReconstructionF32`, the plan and workspace MUST be
`PyramidRowReconstructionPlan<f32>` and
`PyramidRowReconstructionWorkspace<f32>`. The declaration MUST represent one
calibrated F32 row-block input and three conditional outputs: pupil-major
normalized reconstruction pixels, current-frame mean pupil intensity, and the
reconstructed vector. The typed state machine MUST retain HEART-compatible
previous-frame normalization, readiness from the exact pupil origins,
pupil-major coordinate identity, private frame state, finite checks, atomic
terminal publication of all three outputs, and abandonment. A native adapter
MAY replace only reconstruction accumulation for the bounded list of newly
ready pupil-major normalized-pixel ranges from one detector block. It MUST
shard reconstructed output coordinates, MUST NOT parallelize or reorder the
scientific mean, and MUST preserve the serial progressive arrival-order column
sequence for every reconstructed output. Multiple ready ranges from one input
block MUST use one executor rendezvous.

Both progressive operation adapters MUST return before the borrowed input batch
or private accumulator can be reused. A worker failure MUST abandon the
partial frame, publish no output, preserve committed state, and mark the next
complete result discontinuous. An adapter without the required synchronous
executor MUST call the ordinary serial declaration. A synchronous per-block
fork/join implementation MUST NOT claim asynchronous multi-batch overlap.
The multi-lane result MUST be bitwise equal to the serial progressive plan on
the same target; equivalence to a differently grouped complete-frame GEMV MAY
use an explicitly stated tolerance.

Worker count and operating policy belong to graph deployment configuration. A
deployment without a supported executor MUST use the serial declaration rather
than reject an otherwise valid scientific algorithm. Rust and Julia bindings
that expose the same operation marker MUST give it the same mathematical
meaning even when their native serial storage and implementation differ.

Verification intent: compare direct `PreparedGemv<f32>`, native declared
processing, one-lane C execution, and multi-lane FGN execution for the
reconstructor; exercise incremental reconstruction and progressive pyramid plans through native logical
shards, the generated adapter, and real C helper threads; inject executor
failure; inspect both declaration diffs and prove they contain no
worker-specific code.

### CAL-DECL-016 — Progressive detector-readout blocks

A progressive detector-readout declaration MUST consume complete immutable
blocks with common shape `[readout_regions, block_lines, region_samples]`.
All extents MUST be positive. The prepared detector readout mapping MUST fix
the detector extent, total lines per region, block lines, region origins,
signed detector-coordinate line and sample steps, expected acquisition
ordinals, and terminal condition. Preparation MUST prove that the mapped
coordinates form a complete non-overlapping detector image and MUST reject
out-of-range coordinates, duplicate coordinates, unmapped coordinates,
overflow, zero coordinate steps, incompatible region extents, and a block-line
extent that does not divide the prepared lines per region.

Every block MUST carry transport-neutral detector-frame sequence, acquisition
ordinal, terminal, corruption, and discontinuity information. Ordinal zero
MUST begin a frame. Later ordinals MUST be contiguous, use the same sequence,
and MUST NOT assert discontinuity. The terminal indication MUST agree exactly
with the last prepared ordinal. A missing start, changed sequence, skipped or
duplicate ordinal, invalid terminal indication, mid-frame discontinuity, or
corruption MUST abandon the partial frame without publishing output and MUST
mark the next completed output discontinuous.

Pixel calibration MUST prepare complete detector-plane flat and background
values into acquisition order outside processing and MUST evaluate the
established calibration equation over each admitted block without evaluating
the detector-coordinate equation per pixel. A conforming frame assembler MUST
scatter each block into one bounded private detector workspace and MUST publish
one complete detector image only after terminal validation. The single-region
top-to-bottom mapping MUST remain numerically equivalent to the maintained
row-block path. Repeated block processing MUST meet CAL-DECL-007.

Verification intent: compare calibration and complete-frame assembly with the
complete detector-image operations for a single row region, two opposed
regions, and eight regions; cover reversed and transposed coordinate steps;
reject every invalid mapping and acquisition transition before caller-owned
output mutation; and check warmed processing for allocation and deallocation.

### CAL-DECL-017 — Progressive detector-readout Shack–Hartmann reconstruction

The progressive detector-readout Shack–Hartmann composition MUST combine one
validated detector readout mapping with the canonical Shack–Hartmann
scientific geometry. Preparation MUST derive one scientific readiness schedule
that assigns every subaperture to the first acquisition ordinal at which all of
its detector pixels are available. Subaperture declaration order MUST continue
to define pair-interleaved wavefront-slope identity and canonical reconstructor
column order; detector acquisition order MUST NOT redefine that basis.

The composition MUST own bounded private detector, measurement, validity,
flux, and reconstructed-output workspace. It MUST scatter each admitted block,
measure every newly ready subaperture exactly once, and accumulate only the
corresponding canonical reconstructor columns. Prepared reconstruction-plan
construction MUST pack those columns into deterministic acquisition groups
outside processing. The scientist-facing reconstructor MUST remain a complete
row-major matrix in canonical scientific order and MUST NOT contain detector
readout regions, acquisition ordinals, or an execution-specific layout.

The composition MUST publish one reconstructed result only after a complete
valid terminal block. Detector-readout admission failure, scientific execution
failure, non-finite output, or compatible plan adoption during a partial frame
MUST abandon the private state and preserve the previously published result.
Progressive accumulation order MUST be deterministic. It MUST be bitwise equal
to the serial progressive reference on one target; comparison with a
differently grouped complete-frame GEMV MAY use an explicitly stated
tolerance. Repeated block processing MUST meet CAL-DECL-007.

Verification intent: prove complete-frame scientific equivalence for
single-region row order, two opposed regions, and eight-region fixtures; use a
subaperture declaration order different from readiness order; inspect the
prepared matrix grouping; exercise every abandonment condition; and verify
that no camera-readout property enters the canonical reconstructor artifact.

### CAL-DECL-018 — Progressive detector-readout pyramid reconstruction

The progressive detector-readout pyramid composition MUST combine one
validated detector readout mapping with exactly four prepared pupil regions
and the established pupil mask and normalization contract. Preparation MUST
derive which selected pupil pixels become complete at each acquisition
ordinal while preserving pupil-major scientific identity and declaration-order
mean accumulation. Detector readout regions MUST NOT become pupil identities.

The composition MUST preserve CAL-DECL-014 previous-frame normalization,
history adoption, terminal atomic publication, and abandonment semantics. It
MUST process each newly ready selected pupil pixel exactly once and MUST pack
the corresponding canonical reconstructor columns outside processing. A
documented final common linear scale MAY be applied at terminal completion;
an incomplete nonlinear global dependency MUST delay the affected scientific
coordinates rather than assume linearity. Repeated block processing MUST meet
CAL-DECL-007.

Verification intent: compare two consecutive complete results with the
complete-image pyramid path for one-, two-, and eight-region fixtures; prove
previous-frame normalization and declaration-order mean behavior; use pupil
and detector-region orders that differ; cover abandonment and replacement; and
compare canonical and packed reconstruction under the progressive numerical
contract.

## Proof sequence

The first vertical slice is the stateful Float32 leaky integrator. It covers
typed construction, one input, one output, runtime scalar properties, reset,
native execution, FGN execution, and cross-language comparison.

The image-view pattern is also proved independently for PWFS normalization:
one detector image input, four borrowed pupil regions, a shape-preserving pupil
mask parameter, reconstruction-pixel and mean-intensity outputs, native/direct
equivalence, and execution through the C graph host. This is an additional
sensor declaration proof, not a claim that the complete PWFS controller is
already deployed.

The row-block PWFS proof composes that typed normalization with the typed
reconstructor because a conditionally published graph edge cannot expose its
private normalized-pixel prefixes before terminal completion. The composite
does not duplicate either scientific equation: it delegates readiness and
normalization to `PyramidPixelRowPlan` and delegates ordered matrix-column
accumulation to the prepared reconstructor. Its declaration remains directly
usable without FGN, while the generated adapter may shard only the latter
operation.

The corresponding row-block proof consumes calibrated detector rows, advances
every complete row extent within each pupil even when an earlier pupil-major
coordinate is not ready, and publishes the reconstruction pixels and mean
intensity together at frame completion. The scientific mean still uses
declaration-order reduction. It
proves zero-gain startup, previous-frame gain adoption, recovery after a
mid-frame parameter update, warmed allocation-free execution, and direct C
graph-host execution on a small deterministic out-of-order geometry. A
production PWFS profile and scheduled live-device qualification remain
separate work.

The second migration slice is Float32 closed-loop correction as one declaration.
Leaky integration, delayed constraint-feedback anti-windup, hidden-mode
removal, correction output, and controller-state commit remain one scientific
operation. Residual error and the previously realized constraint feedback are
per-frame inputs; the first call ignores the feedback buffer because no prior
correction exists. The two hidden-mode projections are sparse ndarray
parameters, and the four controller gains are runtime scalar properties. This
slice proves the atomic node and its generated adapter; it does not claim that
the current graph host closes feedback cycles or schedules multiple rates.

The third migration slice declares the four stateless deformable-mirror
projection operations: controller-to-VDM, VDM-to-PDM,
PDM-constraint-feedback-to-VDM, and VDM-constraint-feedback-to-controller.
Dimensions are graph-rebuild configuration, frame vectors are data ports, and
the six large projection matrices are sparse ndarray parameters. Concatenated
row or column blocks MUST produce the same result as executing the existing
plans once per controller or physical deformable mirror. The maintained proof
covers multiple controllers and multiple mirrors natively, runs the five-node
deformable-mirror graph at 277 physical actuators through the C host, and loads
that graph through a live PipeWire module.

The next atomic migrations exercise generic controller and calibration
boundaries. Pseudo-open-loop reconstruction remains one two-input, two-output
operation; its four data schemas are graph-rebuild bindings and its three
calibrated matrices are parameter ports. Interaction-matrix stability emits
its complete typed classification and diagnostics. Its two relative limits
are dimensionless runtime properties. Its absolute minimum column norm is
construction configuration because its unit is derived from the configured
interaction-matrix schema and therefore cannot truthfully use a static scalar
property unit.

Matched calibration differencing remains the existing consecutive first-
difference operation. It consumes measurements and their already
exposure-associated corrector sample, publishes both differences plus one
validity flag, and resets its retained previous pair at the lifecycle reset
boundary. The declaration MUST NOT invent the polarity input that appeared in
the earlier adapter catalog because the scientific plan does not consume or
validate polarity.

The structured interaction-matrix decoders preserve their existing batch
boundary: one data input contains a complete ordered response cycle and one
call produces the interaction matrix. Their declarations do not invent
per-frame polarity or pattern-index inputs. A future direct streaming graph
requires a separately specified bounded cycle collector; the present decoder
declarations do not claim that capability.

The optical-gain chain exposes synchronization as typed data rather than an
untyped profile. The dither source publishes its commanded modal vector,
zero-based estimate-window frame, per-mode phase indices, and window-completion
flag. The portable estimator declaration MUST validate all three schedule
signals against its prepared schedule before committing a sample. A skipped
frame or incompatible period therefore fails before estimate publication and
requires a coordinated graph reset. Calibration responses are sparse ndarray
parameters; dither timing remains graph-rebuild configuration because it
defines carried state and coordinated port behavior.

Optical-gain acceptance criteria and reconstructed-mode assignments form one
update policy. Their extents determine the update node's port formats, and a
partial live replacement could combine incompatible criteria and assignments.
The portable update declaration MUST therefore treat the complete policy as
graph-rebuild configuration until the host provides a coherent multi-port
parameter transaction. Lifecycle reset restores the configured initial
physical gains without allocating or reconstructing the workspace.

This projection slice does not replace `DeformableMirrorCommandPlan`. The
existing fused operation rolls back PDM state when a downstream projection
fails. The graph currently has no graph-wide rollback, explicit feedback-delay
primitive, multi-rate scheduling, or atomic transaction spanning several
nodes and parameter ports. A deployment MUST therefore initialize related
matrices while processing is inactive, or accept intermediate mixed
generations, until a higher-level coherent update mechanism is specified.
