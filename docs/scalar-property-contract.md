# FilterGraph scalar property contract

Status: active

Review date: 2026-08-25


Scope update (2026-10-25): this repository delivers Rust implementations only.
References to Julia describe comparison with the external JFG implementation,
not an in-repository package or runtime dependency. Stable requirement IDs and
scientific contracts are retained.

## Authority and scope

This document is the normative authority for transport-independent FilterGraph
scalar properties. Uppercase requirement terms use the meanings defined by
BCP 14 (RFC 2119 and RFC 8174). The numerical algorithm interface remains the
authority for plan, workspace, input, output, and processing semantics.

A property is a low-rate scalar control-plane value. Configuration is input to
instance construction. A parameter port is a sparse large-array update at a
deployment boundary. Per-sample inputs and outputs remain algorithm data.

This contract standardizes behavior shared by Rust, Julia, and deployment
adapters. It does not standardize a scheduler or require PipeWire. The FGN C
ABI and graph-host concurrency contract remain authoritative in the
ndarray-filter-graph repository.

## Requirements

### CAL-PROP-001 — Optional descriptor surface

An algorithm with scalar properties MUST enumerate stable descriptor-local
identifiers and canonical unqualified names, and each descriptor must declare
its value type, default, non-empty description, canonical non-empty unit,
update class, and any range or named choices. An algorithm without properties
remains usable through the base algorithm interface without property methods.

Verification intent: enumerate descriptors in Rust and Julia and verify the
metadata and empty-interface behavior.

### CAL-PROP-002 — Scalar value domain

The common property value domain MUST represent Boolean, signed 32-bit and
64-bit integer, IEEE 754 binary32 and binary64, unsigned 32-bit identifier, and
UTF-8 string values. Large arrays, reconstructors, masks, reference vectors,
and calibration artifacts are parameter-port or prepared-plan data, not
properties.

Verification intent: construct and distinguish every common scalar kind and
inspect adapter schemas for array-valued properties.

### CAL-PROP-003 — Update classes

Every property MUST declare exactly one of read-only observation,
runtime-writable scalar, construction-only configuration, or construction
configuration whose change requires graph reconstruction. A runtime
transaction rejects all assignments except runtime-writable scalars without
changing requested or active state.

Verification intent: exercise each rejected class and observe unchanged plans
and generations.

### CAL-PROP-004 — Transactional preparation

Property preparation MUST validate a complete local assignment set and produce
a logically read-only replacement plan. Duplicate or unknown identifiers,
incompatible scalar kinds, descriptor-constraint violations, and invalid
cross-property combinations reject the complete transaction while preserving
active and requested plans and generations.

Verification intent: combine valid and invalid assignments and verify
all-or-nothing behavior in Rust, Julia, and the FGN graph host.

### CAL-PROP-005 — Plan and workspace compatibility

A runtime-writable update MUST preserve the concrete algorithm, input, output,
and workspace compatibility contract. A value that changes a port format,
storage extent, backend, or workspace representation is construction-only or
graph-reconstruction configuration.

Verification intent: use the same caller-owned buffers and workspace before
and after an admitted runtime update.

### CAL-PROP-006 — Bounded publication and adoption

A portable binding or deployment adapter MUST bound staged and requested
property state, report back pressure without replacing an earlier request when
capacity is unavailable, publish successful preparation without failure, and
adopt the complete requested plan immediately before one complete process
boundary. Partial plan adoption is forbidden.

Verification intent: fill the update capacity, observe busy status, process one
sample, and then observe capacity recovery and complete-plan adoption.

### CAL-PROP-007 — Requested and active generations

A binding MUST expose monotonic requested and active generations. Portable Rust
and Julia bindings use unsigned 64-bit storage. The FGN adapter exposes
non-negative `SPA_TYPE_Long` values and therefore reports exhaustion at
`typemax(Int64)`. Publishing one non-empty transaction advances requested
generation exactly once; process-boundary adoption sets active generation to
that value; preparation, discard, rejection, and a process without a request do
not advance either generation. Generation exhaustion reports an error before
publication.

Verification intent: inspect both generations after prepare, commit, discard,
rejection, adoption, unchanged processing, and exhaustion.

### CAL-PROP-008 — Repeated-path constraints

An admitted real-time implementation MUST perform property adoption and
subsequent algorithm processing with bounded work and without allocation,
blocking, locks, destruction of the retired plan, or unwinding on the data-loop
path. Parsing, validation, allocation, plan preparation, and retired-plan
reclamation occur off that path.

Verification intent: allocation tests for the Julia and Rust process boundary,
plus C/Rust FGN ABI inspection and integration tests.

### CAL-PROP-009 — Change observation

A deployment adapter that exposes properties to an external control plane MUST
provide a bounded indication when a requested or active property value changes,
without calling an external notification callback from the numerical process
path. High-rate measurements and per-sample validity remain outputs or
metadata.

Verification intent: adopt one update, observe one process result flag, and
publish the property snapshot outside the data loop.

### CAL-PROP-010 — Cross-language conformance

Rust and Julia implementations of the same algorithm property surface MUST use
the same identifiers, names, units, validation rules, update classes,
numerical results, and generation transitions. A C ABI adapter may use a
different storage or synchronization mechanism only when it preserves those
observable semantics.

Verification intent: run the shared leaky-integrator fixture in Rust and Julia
and run the FilterGraph-backed FGN plugin through the C graph-host test.

### CAL-PROP-011 — Declarative algorithm authoring

A property-enabled Rust or Julia algorithm MUST add only a declarative schema,
a typed mapping from plan to current property values, and one replacement-plan
constructor to its numerical interface. Shared infrastructure owns schema and
assignment validation, lookup, transaction iteration, generation tracking,
bounded publication, and adoption. C ABI machinery remains adapter code.

Verification intent: inspect the Rust and Julia leaky-integrator implementations
and run their negative, conformance, generation, and allocation tests.

### CAL-PROP-012 — Schema admission and evolution

An algorithm property schema MUST be rejected before deployment when descriptor
IDs or names are duplicated, required text is empty, a default or constraint
uses an incompatible scalar kind, a range is unordered or unsupported, a
default violates its constraint, or choice names or values are duplicated.
Once published, an ID and canonical name are not reused for another meaning;
incompatible schema evolution uses a new algorithm-interface version.

Portable ordered ranges are limited to signed integers, floating-point values,
and unsigned IDs. Boolean and string properties use no constraint or named
choices.

Verification intent: admit malformed schemas in Rust, Julia, and C and verify a
structured or errno-style rejection before processing.

### CAL-PROP-013 — Empty and equivalent transactions

An empty assignment set MUST be a successful no-op that occupies no staging
capacity and advances no generation. A non-empty valid assignment set is a
request even when every assigned value equals its current value and advances
requested generation once when published.

Verification intent: prepare and commit empty and same-value transactions and
inspect capacity and generation transitions.

### CAL-PROP-014 — Configuration and observation ownership

The algorithm plan MUST remain the authority for scientific scalar
configuration, while a deployment binding owns dynamic requested/active
generations, runtime status, and transport observations. A binding may expose
those observations as read-only properties without adding them to the
algorithm plan schema.

Verification intent: compare plan descriptor enumeration with an adapter's
augmented read-only surface and verify that plan lookup never depends on
transport state.

### CAL-PROP-015 — FGN adapter projection

A FilterGraph FGN adapter MUST project algorithm descriptor metadata, current plan
values, and complete property transactions to the FGN ABI without requiring an
algorithm-specific ABI implementation. Scalar plan properties map to FGN
properties, large prepared arrays map to parameter ports, and per-frame values
map to data ports. The adapter supplies the concurrency, lifetime, generation,
and error translation required by the FGN contract.
The low-level Rust `PluginInstance` extension is unsafe and requires
`Send + Sync`;
scientist-facing declarations use safe macros backed by adapter-owned
single-data-owner workspaces. Output adapters preserve the host offset and
publish a completed size only after successful numerical processing.

Verification intent: export a FilterGraph algorithm through shared adapter code,
load it through the C graph host, update multiple properties together, and
verify numerical and generation results against the shared fixture.

## Scientist-facing FGN deployment

### CAL-FGN-001 — Scientist and deployment boundary

A Rust scientist authoring an algorithm for later FGN deployment MUST need to
implement only an ordinary transport-neutral typed operation and one portable
FilterGraph declaration beside it. The declaration supplies the stable label,
typed construction, scientific ports, schemas, property policy, sparse
parameter replacement, preparation, processing, and reset in ordinary Rust.
The scientific implementation and declaration MUST NOT parse PipeWire
configuration or use FGN or SPA ABI structures, raw pointers, C strings,
errno values, manual unwind containment, buffer conversion, callback tables,
or real-time publication machinery. Implementing `AlgorithmPlan` is optional.

Shared adapter code MUST derive all transport, descriptor, and lifecycle
behavior from that declaration. Deployment MUST NOT require a separate
algorithm-specific binding or a change to the scientific operation.

Construction configuration MUST establish a valid initial plan. Per-frame
values use data ports, sparse large-array replacements use parameter ports,
runtime-writable low-rate scalars use properties, and a change to port shape or
graph topology requires graph reconstruction.

Verification intent: deploy property-enabled, property-free, arbitrary-port,
and transport-neutral composite FilterGraph plans; inspect the scientific modules
for prohibited ABI machinery; exercise construction configuration and sparse
parameter replacement through the exported descriptor.

### CAL-FGN-002 — Multi-descriptor plugin registry

One FilterGraph FGN shared library MUST be able to expose multiple algorithm
descriptors under one plugin-family interface. Exact lookup of every registered
unique label returns its stable descriptor; null and unknown labels return no
descriptor. Duplicate or empty registered labels make the plugin-family
interface unavailable rather than selecting an ambiguous descriptor.

Verification intent: register at least two real FilterGraph algorithms, resolve
and execute both through the exported shared-library entry point, and reject
null, unknown, empty, and duplicate-label registries.

### CAL-FGN-003 — Complete Rust production selection

The maintained Rust production shared library MUST select every portable
declaration admitted for production deployment. The selection MUST contain
declaration types only. Port roles, scalar properties, construction,
numerical types, shapes, schemas, and callbacks MUST be derived from each local
declaration and MUST NOT be repeated in a deployment catalog. Large arrays
remain parameter ports under CAL-PROP-002 and CAL-PROP-015.

An algorithm that is intentionally kept atomic, or is not yet safe under the
host's scheduling and failure semantics, MUST document that disposition at
the scientific operation boundary. It need not acquire a placeholder adapter
entry. Adding an admitted algorithm MAY require a deployment maintainer to add
its declaration type to a particular shared-library selection; it MUST NOT
require a scientist to write a registry entry or adapter implementation.

Verification intent: inspect the type-only production selection, compare its
size with the admitted declaration inventory, reject duplicate labels, audit
the dynamic export surface, stage an installation, and validate selected
descriptors through the FGN host.

### CAL-FGN-004 — Unified property and parameter publication

A maintained deployment for an algorithm with both runtime scalar properties
and sparse ndarray parameters MUST prepare both update kinds against the same
active immutable plan and publish replacements through one bounded plan owner.
At most one requested replacement of either kind may await adoption. A scalar
property replacement preserves the active parameter artifacts, a parameter
replacement preserves the active scalar property values, and an unavailable
slot reports back pressure without displacing an earlier replacement. Each
successful non-empty update advances only its corresponding requested and
active generation or sequence, and the complete replacement is adopted
immediately before one complete process boundary. This deployment support MUST
NOT require a change to the scientific `AlgorithmPlan` implementation.

Verification intent: interleave scalar-property and ndarray-parameter
preparation, publication, rejection, discard, and process-boundary adoption;
inspect both observation pairs and execute the resulting plan without warmed
allocation or retired-plan destruction.

### CAL-FGN-005 — Declarative ndarray parameter transactions

A parameter-capable portable declaration MUST be able to prepare one complete
replacement plan from a non-empty set of unique parameter-port assignments.
The shared default MUST compose assignments in supplied order against a private
replacement and install nothing if any assignment fails. A scientific
declaration MAY override that method to validate coupled artifacts together,
but MUST NOT use FGN, SPA, raw pointers, callback tokens, or errno values.

The generated FGN adapter MUST expose the batch callback automatically for a
parameter-capable declaration, validate every local port and borrowed typed
view, require all present Header sequences in one local transaction to agree,
prepare one publication token, and adopt it at the graph-cycle boundary. A
declaration without parameter ports MUST expose no batch callback. None of this
requires an algorithm-specific deployment wrapper or registry entry.

Verification intent: replace flat and background calibration planes together
without an intermediate mixed plan, reject duplicate and invalid assignments
without changing the active plan, and update two linked FilterGraph nodes in one
graph cycle through the C host.

## Error and lifecycle model

Schema admission and transaction preparation are control-path operations and
may return rich language errors. An FGN adapter maps them to documented
negative errno values. Publication is non-failing after successful preparation.
The previous active plan remains owned until a control context can reclaim it.

Construction-only properties are accepted only while creating an instance.
Graph-rebuild properties are not applied to a live instance; the deployment
owner builds and validates a replacement graph and then performs its own graph
activation transition.

## Standard SPA projection

The FGN ABI is the complete property representation. Standard
`SPA_PARAM_PropInfo` and `SPA_PARAM_Props` are a deliberately lossy PipeWire
view: they do not preserve canonical units, local IDs, or the distinction among
read-only, construction-only, and graph-rebuild values. Standard Params are
sufficient for runtime scalar control; clients that need the complete schema
use the FGN descriptor surface or a future explicit schema parameter.
