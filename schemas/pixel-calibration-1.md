# Pixel calibration schemas, version 1

These schemas define the scientific ndarray values used by detector pixel
calibration. PipeWireAO owns generic ndarray structural fields; the pinned
public catalog defines the scientific identifiers and meanings below.

Pixel calibration evaluates each detector pixel in this exact order:

```text
calibrated[i] = flat[i] * (f32(raw[i]) - background[i])
```

The Rust ndarray declaration accepts detector ADU with schema
`pipewireao/raw-detector-pixels-u16/1`. Conversion from a native video format
belongs to the external acquisition boundary.

## `pipewireao/flat-calibration-f32/1`

- Scientific quantity: complete detector flat-calibration coefficient.
- Unit: PDE/ADU.
- Structural format: `F32_LE`, shape `[height, width]`, `ROW_MAJOR`.
- Update role: sparse prepared artifact, not a per-frame input.
- Validity: every coefficient is finite.

## `pipewireao/background-calibration-f32/1`

- Scientific quantity: detector background calibration.
- Unit: detector ADU.
- Structural format: `F32_LE`, shape `[height, width]`, `ROW_MAJOR`.
- Update role: sparse prepared artifact, not a per-frame input.
- Validity: every value is finite.

## `pipewireao/calibrated-pixels-f32/1`

- Scientific quantity: calibrated detector pixels.
- Unit: PDE.
- Structural format: `F32_LE`, shape `[height, width]`, `ROW_MAJOR`.
- Rate: present and equal to the accepted raw detector-frame rate.
- Ordering: detector row-major order, with column varying fastest.

The ndarray format has no node-wide interpretation profile. Detector-coordinate
identity and calibration provenance are bound to the algorithm instance by its
construction and artifact-loading configuration; equal element type, shape,
layout, and schema do not prove that two artifacts came from the same detector.
A deployment that needs graph-level enforcement must first specify a typed,
versioned per-port geometry or provenance field and its update lifecycle. It
must not reuse one untyped node-wide string for this purpose.

Missing flat and background artifacts select the unit flat and zero
background respectively. Replacing either artifact does not change the schema
identifier. A paired replacement becomes active atomically at a complete-frame
boundary.
