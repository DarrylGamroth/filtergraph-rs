//! Prepared multi-region detector readout mapping and frame assembly.

use alloc::{boxed::Box, vec, vec::Vec};
use core::{fmt, mem};

/// One prepared detector readout region.
///
/// Coordinates and steps use `(row, column)` order. Steps describe movement
/// through detector coordinates and are independent of ndarray memory stride.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorReadoutRegion {
    origin: (usize, usize),
    line_step: (isize, isize),
    sample_step: (isize, isize),
}

impl DetectorReadoutRegion {
    /// Defines one detector readout region from its origin and signed steps.
    pub const fn new(
        origin: (usize, usize),
        line_step: (isize, isize),
        sample_step: (isize, isize),
    ) -> Self {
        Self {
            origin,
            line_step,
            sample_step,
        }
    }

    /// Returns the zero-based detector origin in `(row, column)` order.
    pub const fn origin(&self) -> (usize, usize) {
        self.origin
    }

    /// Returns the signed detector-coordinate step between readout lines.
    pub const fn line_step(&self) -> (isize, isize) {
        self.line_step
    }

    /// Returns the signed detector-coordinate step between samples.
    pub const fn sample_step(&self) -> (isize, isize) {
        self.sample_step
    }
}

/// Invalid prepared detector readout mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorReadoutError {
    /// A detector or readout extent is zero.
    EmptyExtent,
    /// A product or coordinate calculation overflowed.
    ExtentOverflow,
    /// Block lines do not divide the total lines in each region.
    BlockLines,
    /// The region geometry does not contain exactly one value per detector pixel.
    ImageExtent {
        /// Number of pixels described by the readout regions.
        readout: usize,
        /// Number of pixels in the detector image.
        detector: usize,
    },
    /// A region has a zero detector-coordinate line step.
    ZeroLineStep {
        /// Zero-based readout-region index.
        region: usize,
    },
    /// A region has a zero detector-coordinate sample step.
    ZeroSampleStep {
        /// Zero-based readout-region index.
        region: usize,
    },
    /// A mapped coordinate falls outside the detector image.
    CoordinateRange {
        /// Zero-based readout-region index.
        region: usize,
    },
    /// Two readout samples map to the same detector coordinate.
    DuplicateCoordinate {
        /// Zero-based detector row.
        row: usize,
        /// Zero-based detector column.
        column: usize,
    },
    /// No readout sample maps to one detector coordinate.
    UnmappedCoordinate {
        /// Zero-based detector row.
        row: usize,
        /// Zero-based detector column.
        column: usize,
    },
}

impl fmt::Display for DetectorReadoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyExtent => formatter.write_str("detector readout extents must be positive"),
            Self::ExtentOverflow => formatter.write_str("detector readout extent overflow"),
            Self::BlockLines => formatter
                .write_str("readout block lines must divide the total lines in each region"),
            Self::ImageExtent { readout, detector } => write!(
                formatter,
                "readout pixel extent {readout} does not match detector extent {detector}"
            ),
            Self::ZeroLineStep { region } => {
                write!(formatter, "readout region {region} has a zero line step")
            }
            Self::ZeroSampleStep { region } => {
                write!(formatter, "readout region {region} has a zero sample step")
            }
            Self::CoordinateRange { region } => {
                write!(
                    formatter,
                    "readout region {region} exceeds the detector image"
                )
            }
            Self::DuplicateCoordinate { row, column } => write!(
                formatter,
                "detector coordinate ({row}, {column}) is mapped more than once"
            ),
            Self::UnmappedCoordinate { row, column } => {
                write!(
                    formatter,
                    "detector coordinate ({row}, {column}) is not mapped"
                )
            }
        }
    }
}

impl core::error::Error for DetectorReadoutError {}

/// Invalid portable detector-readout profile fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorReadoutProfileError {
    /// Origin, line-step, and sample-step arrays have different extents.
    RegionExtent {
        /// Number of detector origins.
        origins: usize,
        /// Number of detector-coordinate line steps.
        line_steps: usize,
        /// Number of detector-coordinate sample steps.
        sample_steps: usize,
    },
    /// The prepared detector-coordinate mapping is invalid.
    Mapping(DetectorReadoutError),
}

impl fmt::Display for DetectorReadoutProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegionExtent {
                origins,
                line_steps,
                sample_steps,
            } => write!(
                formatter,
                "detector readout profile has {origins} origins, {line_steps} line steps, and {sample_steps} sample steps"
            ),
            Self::Mapping(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for DetectorReadoutProfileError {}

impl From<DetectorReadoutError> for DetectorReadoutProfileError {
    fn from(error: DetectorReadoutError) -> Self {
        Self::Mapping(error)
    }
}

/// Invalid detector or readout-block array supplied to a prepared mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorReadoutDataError {
    /// The acquisition ordinal is outside the prepared frame.
    OrdinalRange {
        /// Supplied acquisition ordinal.
        ordinal: usize,
        /// Prepared number of blocks per detector frame.
        blocks: usize,
    },
    /// A detector image has the wrong element extent.
    DetectorExtent {
        /// Required detector-image extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// A detector readout block has the wrong element extent.
    BlockExtent {
        /// Required block extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
}

impl fmt::Display for DetectorReadoutDataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrdinalRange { ordinal, blocks } => write!(
                formatter,
                "acquisition ordinal {ordinal} is outside {blocks} prepared blocks"
            ),
            Self::DetectorExtent { expected, actual } => write!(
                formatter,
                "detector image extent {actual} does not match {expected}"
            ),
            Self::BlockExtent { expected, actual } => write!(
                formatter,
                "detector readout block extent {actual} does not match {expected}"
            ),
        }
    }
}

impl core::error::Error for DetectorReadoutDataError {}

/// Prepared mapping from acquisition-ordered readout blocks to detector coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectorReadoutPlan {
    detector_size: (usize, usize),
    region_lines: usize,
    block_lines: usize,
    region_samples: usize,
    regions: Box<[DetectorReadoutRegion]>,
    blocks: usize,
    block_elements: usize,
    detector_indices: Box<[usize]>,
    arrival_ordinals: Box<[usize]>,
}

impl DetectorReadoutPlan {
    /// Builds a mapping from portable signed 32-bit coordinate-step arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn from_i32_coordinate_steps(
        detector_size: (usize, usize),
        region_lines: usize,
        block_lines: usize,
        region_samples: usize,
        origins: &[[usize; 2]],
        line_steps: &[[i32; 2]],
        sample_steps: &[[i32; 2]],
    ) -> Result<Self, DetectorReadoutProfileError> {
        if origins.len() != line_steps.len() || origins.len() != sample_steps.len() {
            return Err(DetectorReadoutProfileError::RegionExtent {
                origins: origins.len(),
                line_steps: line_steps.len(),
                sample_steps: sample_steps.len(),
            });
        }
        let regions = origins
            .iter()
            .zip(line_steps)
            .zip(sample_steps)
            .map(|((origin, line_step), sample_step)| {
                DetectorReadoutRegion::new(
                    (origin[0], origin[1]),
                    (line_step[0] as isize, line_step[1] as isize),
                    (sample_step[0] as isize, sample_step[1] as isize),
                )
            })
            .collect::<Vec<_>>();
        Self::new(
            detector_size,
            region_lines,
            block_lines,
            region_samples,
            regions,
        )
        .map_err(Into::into)
    }

    /// Validates and prepares one complete non-overlapping detector mapping.
    pub fn new(
        detector_size: (usize, usize),
        region_lines: usize,
        block_lines: usize,
        region_samples: usize,
        regions: impl Into<Box<[DetectorReadoutRegion]>>,
    ) -> Result<Self, DetectorReadoutError> {
        let regions = regions.into();
        let (detector_rows, detector_columns) = detector_size;
        if detector_rows == 0
            || detector_columns == 0
            || region_lines == 0
            || block_lines == 0
            || region_samples == 0
            || regions.is_empty()
        {
            return Err(DetectorReadoutError::EmptyExtent);
        }
        if block_lines > region_lines || !region_lines.is_multiple_of(block_lines) {
            return Err(DetectorReadoutError::BlockLines);
        }
        let detector_elements = detector_rows
            .checked_mul(detector_columns)
            .ok_or(DetectorReadoutError::ExtentOverflow)?;
        let readout_elements = regions
            .len()
            .checked_mul(region_lines)
            .and_then(|extent| extent.checked_mul(region_samples))
            .ok_or(DetectorReadoutError::ExtentOverflow)?;
        if readout_elements != detector_elements {
            return Err(DetectorReadoutError::ImageExtent {
                readout: readout_elements,
                detector: detector_elements,
            });
        }
        let blocks = region_lines / block_lines;
        let block_elements = regions
            .len()
            .checked_mul(block_lines)
            .and_then(|extent| extent.checked_mul(region_samples))
            .ok_or(DetectorReadoutError::ExtentOverflow)?;
        let mut detector_indices = Vec::with_capacity(detector_elements);
        let mut arrival_ordinals = vec![usize::MAX; detector_elements];

        for (region_index, region) in regions.iter().enumerate() {
            if region.line_step == (0, 0) {
                return Err(DetectorReadoutError::ZeroLineStep {
                    region: region_index,
                });
            }
            if region.sample_step == (0, 0) {
                return Err(DetectorReadoutError::ZeroSampleStep {
                    region: region_index,
                });
            }
        }

        for ordinal in 0..blocks {
            for (region_index, region) in regions.iter().enumerate() {
                for local_line in 0..block_lines {
                    let line = ordinal
                        .checked_mul(block_lines)
                        .and_then(|first| first.checked_add(local_line))
                        .ok_or(DetectorReadoutError::ExtentOverflow)?;
                    for sample in 0..region_samples {
                        let row = coordinate(
                            region.origin.0,
                            line,
                            region.line_step.0,
                            sample,
                            region.sample_step.0,
                        )
                        .ok_or(DetectorReadoutError::CoordinateRange {
                            region: region_index,
                        })?;
                        let column = coordinate(
                            region.origin.1,
                            line,
                            region.line_step.1,
                            sample,
                            region.sample_step.1,
                        )
                        .ok_or(DetectorReadoutError::CoordinateRange {
                            region: region_index,
                        })?;
                        if row >= detector_rows || column >= detector_columns {
                            return Err(DetectorReadoutError::CoordinateRange {
                                region: region_index,
                            });
                        }
                        let detector_index = row
                            .checked_mul(detector_columns)
                            .and_then(|start| start.checked_add(column))
                            .ok_or(DetectorReadoutError::ExtentOverflow)?;
                        if arrival_ordinals[detector_index] != usize::MAX {
                            return Err(DetectorReadoutError::DuplicateCoordinate { row, column });
                        }
                        arrival_ordinals[detector_index] = ordinal;
                        detector_indices.push(detector_index);
                    }
                }
            }
        }
        if let Some(index) = arrival_ordinals
            .iter()
            .position(|&ordinal| ordinal == usize::MAX)
        {
            return Err(DetectorReadoutError::UnmappedCoordinate {
                row: index / detector_columns,
                column: index % detector_columns,
            });
        }

        Ok(Self {
            detector_size,
            region_lines,
            block_lines,
            region_samples,
            regions,
            blocks,
            block_elements,
            detector_indices: detector_indices.into_boxed_slice(),
            arrival_ordinals: arrival_ordinals.into_boxed_slice(),
        })
    }

    /// Returns the complete logical detector size in `(rows, columns)` order.
    pub const fn detector_size(&self) -> (usize, usize) {
        self.detector_size
    }

    /// Returns the number of lines acquired from each region in one frame.
    pub const fn region_lines(&self) -> usize {
        self.region_lines
    }

    /// Returns the number of lines from each region in one readout block.
    pub const fn block_lines(&self) -> usize {
        self.block_lines
    }

    /// Returns the number of samples in each readout-region line.
    pub const fn region_samples(&self) -> usize {
        self.region_samples
    }

    /// Returns the prepared detector readout regions.
    pub fn regions(&self) -> &[DetectorReadoutRegion] {
        &self.regions
    }

    /// Returns the number of readout blocks in one complete detector frame.
    pub const fn blocks(&self) -> usize {
        self.blocks
    }

    /// Returns the element count of one `[region, line, sample]` block.
    pub const fn block_elements(&self) -> usize {
        self.block_elements
    }

    /// Returns the complete detector-image element count.
    pub const fn detector_elements(&self) -> usize {
        self.detector_indices.len()
    }

    /// Returns the acquisition ordinal at which one detector coordinate arrives.
    pub fn arrival_ordinal(&self, row: usize, column: usize) -> Option<usize> {
        let (rows, columns) = self.detector_size;
        if row >= rows || column >= columns {
            return None;
        }
        self.arrival_ordinals.get(row * columns + column).copied()
    }

    /// Returns detector indices for one acquisition block in ndarray element order.
    pub fn block_detector_indices(
        &self,
        ordinal: usize,
    ) -> Result<&[usize], DetectorReadoutDataError> {
        let start = self.block_start(ordinal)?;
        Ok(&self.detector_indices[start..start + self.block_elements])
    }

    /// Copies one complete detector plane into acquisition order during preparation.
    pub fn pack_detector<T: Copy>(
        &self,
        detector: &[T],
    ) -> Result<Box<[T]>, DetectorReadoutDataError> {
        if detector.len() != self.detector_elements() {
            return Err(DetectorReadoutDataError::DetectorExtent {
                expected: self.detector_elements(),
                actual: detector.len(),
            });
        }
        Ok(self
            .detector_indices
            .iter()
            .map(|&index| detector[index])
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Scatters one immutable acquisition block into a detector image.
    pub fn scatter_block<T: Copy>(
        &self,
        detector: &mut [T],
        block: &[T],
        ordinal: usize,
    ) -> Result<(), DetectorReadoutDataError> {
        if detector.len() != self.detector_elements() {
            return Err(DetectorReadoutDataError::DetectorExtent {
                expected: self.detector_elements(),
                actual: detector.len(),
            });
        }
        if block.len() != self.block_elements {
            return Err(DetectorReadoutDataError::BlockExtent {
                expected: self.block_elements,
                actual: block.len(),
            });
        }
        for (&index, &value) in self.block_detector_indices(ordinal)?.iter().zip(block) {
            detector[index] = value;
        }
        Ok(())
    }

    fn block_start(&self, ordinal: usize) -> Result<usize, DetectorReadoutDataError> {
        if ordinal >= self.blocks {
            return Err(DetectorReadoutDataError::OrdinalRange {
                ordinal,
                blocks: self.blocks,
            });
        }
        Ok(ordinal * self.block_elements)
    }
}

fn coordinate(
    origin: usize,
    line: usize,
    line_step: isize,
    sample: usize,
    sample_step: isize,
) -> Option<usize> {
    let value = i128::try_from(origin)
        .ok()?
        .checked_add(i128::try_from(line).ok()?.checked_mul(line_step as i128)?)?
        .checked_add(
            i128::try_from(sample)
                .ok()?
                .checked_mul(sample_step as i128)?,
        )?;
    usize::try_from(value).ok()
}

/// Portable identity and acquisition placement of one detector readout block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorReadoutBlockInfo {
    /// Stable detector-frame identity shared by every block in one frame.
    pub sequence: u64,
    /// Zero-based acquisition ordinal within the detector frame.
    pub ordinal: usize,
    /// Whether this is the final prepared block in the detector frame.
    pub terminal: bool,
    /// Whether this frame follows a loss or abandoned frame.
    pub discontinuity: bool,
    /// Whether acquisition marked this block corrupt.
    pub corrupted: bool,
}

/// Recoverable reason that one partial detector frame was abandoned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorReadoutDropReason {
    /// A non-initial block arrived without an active frame.
    MissingStart,
    /// A block used a different sequence from the active frame.
    Sequence,
    /// The supplied ordinal was not the next required acquisition ordinal.
    AcquisitionOrder,
    /// The supplied ordinal was outside the prepared detector frame.
    OrdinalRange,
    /// The terminal indication did not match the final prepared ordinal.
    Terminal,
    /// Discontinuity was asserted after the first block of a frame.
    MidFrameDiscontinuity,
    /// Acquisition marked the block corrupt.
    Corrupted,
}

/// Publication result after consuming one complete detector readout block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorReadoutResult {
    /// The detector frame remains in progress and output is unchanged.
    Pending,
    /// A complete frame-rate result was written to caller-owned output.
    Complete {
        /// Stable detector-frame identity of the completed result.
        sequence: u64,
        /// Whether the result follows any reported or locally detected loss.
        discontinuity: bool,
    },
    /// The supplied block abandoned a partial frame without publishing output.
    Dropped(DetectorReadoutDropReason),
}

/// Common single-owner acquisition-order state for detector readout blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DetectorReadoutFrame {
    sequence: Option<u64>,
    expected_ordinal: usize,
    pending_discontinuity: bool,
    current_discontinuity: bool,
}

impl DetectorReadoutFrame {
    pub(crate) const fn new() -> Self {
        Self {
            sequence: None,
            expected_ordinal: 0,
            pending_discontinuity: false,
            current_discontinuity: false,
        }
    }

    pub(crate) const fn sequence(&self) -> Option<u64> {
        self.sequence
    }

    pub(crate) const fn discontinuity(&self) -> bool {
        self.current_discontinuity
    }

    pub(crate) fn accept(
        &mut self,
        info: DetectorReadoutBlockInfo,
        blocks: usize,
    ) -> Result<(), DetectorReadoutDropReason> {
        if info.corrupted {
            self.abandon();
            return Err(DetectorReadoutDropReason::Corrupted);
        }
        if info.ordinal >= blocks {
            self.abandon();
            return Err(DetectorReadoutDropReason::OrdinalRange);
        }
        if info.ordinal == 0 {
            let superseded = self.sequence.is_some();
            if superseded {
                self.abandon();
            }
            self.sequence = Some(info.sequence);
            self.expected_ordinal = 0;
            self.current_discontinuity =
                mem::take(&mut self.pending_discontinuity) || info.discontinuity || superseded;
        } else {
            let Some(sequence) = self.sequence else {
                self.abandon();
                return Err(DetectorReadoutDropReason::MissingStart);
            };
            if sequence != info.sequence {
                self.abandon();
                return Err(DetectorReadoutDropReason::Sequence);
            }
            if info.discontinuity {
                self.abandon();
                return Err(DetectorReadoutDropReason::MidFrameDiscontinuity);
            }
        }
        if info.ordinal != self.expected_ordinal {
            self.abandon();
            return Err(DetectorReadoutDropReason::AcquisitionOrder);
        }
        let terminal = info.ordinal + 1 == blocks;
        if info.terminal != terminal {
            self.abandon();
            return Err(DetectorReadoutDropReason::Terminal);
        }
        self.expected_ordinal = info.ordinal + 1;
        Ok(())
    }

    pub(crate) fn abandon(&mut self) {
        self.sequence = None;
        self.expected_ordinal = 0;
        self.pending_discontinuity = true;
        self.current_discontinuity = false;
    }

    pub(crate) fn complete(&mut self) {
        self.sequence = None;
        self.expected_ordinal = 0;
        self.pending_discontinuity = false;
        self.current_discontinuity = false;
    }
}

/// Invalid detector-frame assembly call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorFrameAssemblerError {
    /// The caller-owned complete detector image has the wrong extent.
    OutputExtent {
        /// Required detector-image extent.
        expected: usize,
        /// Supplied output extent.
        actual: usize,
    },
    /// The detector readout block has the wrong extent.
    InputExtent {
        /// Required block extent.
        expected: usize,
        /// Supplied block extent.
        actual: usize,
    },
}

impl fmt::Display for DetectorFrameAssemblerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "detector-frame output extent {actual} does not match {expected}"
            ),
            Self::InputExtent { expected, actual } => write!(
                formatter,
                "detector readout block extent {actual} does not match {expected}"
            ),
        }
    }
}

impl core::error::Error for DetectorFrameAssemblerError {}

/// Prepared progressive assembly of readout blocks into one detector image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectorFrameAssemblerPlan {
    readout: DetectorReadoutPlan,
}

impl DetectorFrameAssemblerPlan {
    /// Retains one validated detector readout mapping.
    pub const fn new(readout: DetectorReadoutPlan) -> Self {
        Self { readout }
    }

    /// Returns the prepared detector readout mapping.
    pub const fn readout(&self) -> &DetectorReadoutPlan {
        &self.readout
    }

    /// Allocates bounded mutable storage for one execution owner.
    pub fn workspace<T>(&self) -> DetectorFrameAssemblerWorkspace<T>
    where
        T: Clone + Default,
    {
        DetectorFrameAssemblerWorkspace {
            detector: vec![T::default(); self.readout.detector_elements()].into_boxed_slice(),
            frame: DetectorReadoutFrame::new(),
        }
    }

    /// Consumes one immutable detector readout block.
    pub fn process_block<T: Copy>(
        &self,
        output: &mut [T],
        workspace: &mut DetectorFrameAssemblerWorkspace<T>,
        block: &[T],
        info: DetectorReadoutBlockInfo,
    ) -> Result<DetectorReadoutResult, DetectorFrameAssemblerError> {
        if output.len() != self.readout.detector_elements() {
            return Err(DetectorFrameAssemblerError::OutputExtent {
                expected: self.readout.detector_elements(),
                actual: output.len(),
            });
        }
        if block.len() != self.readout.block_elements() {
            return Err(DetectorFrameAssemblerError::InputExtent {
                expected: self.readout.block_elements(),
                actual: block.len(),
            });
        }
        if let Err(reason) = workspace.frame.accept(info, self.readout.blocks()) {
            return Ok(DetectorReadoutResult::Dropped(reason));
        }
        self.readout
            .scatter_block(&mut workspace.detector, block, info.ordinal)
            .expect("validated extents and admitted ordinal");
        if !info.terminal {
            return Ok(DetectorReadoutResult::Pending);
        }
        output.copy_from_slice(&workspace.detector);
        let result = DetectorReadoutResult::Complete {
            sequence: info.sequence,
            discontinuity: workspace.frame.discontinuity(),
        };
        workspace.frame.complete();
        Ok(result)
    }
}

/// Bounded mutable storage for one detector-frame assembly owner.
#[derive(Debug)]
pub struct DetectorFrameAssemblerWorkspace<T> {
    detector: Box<[T]>,
    frame: DetectorReadoutFrame,
}

impl<T> DetectorFrameAssemblerWorkspace<T> {
    /// Returns the active detector-frame sequence, if any.
    pub const fn sequence(&self) -> Option<u64> {
        self.frame.sequence()
    }

    /// Abandons any partial detector frame.
    pub fn reset(&mut self) {
        self.frame.abandon();
    }
}

/// Invalid portable detector-frame assembler execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorFrameAssemblerProcessError {
    /// The input block did not carry transport-neutral sample identity.
    MissingSampleMetadata,
    /// The portable acquisition ordinal cannot be represented on this target.
    OrdinalOverflow,
    /// The prepared frame assembler rejected the call.
    Assembly(DetectorFrameAssemblerError),
}

impl fmt::Display for DetectorFrameAssemblerProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => {
                formatter.write_str("detector readout block sample metadata is required")
            }
            Self::OrdinalOverflow => {
                formatter.write_str("detector readout acquisition ordinal does not fit usize")
            }
            Self::Assembly(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for DetectorFrameAssemblerProcessError {}

impl From<DetectorFrameAssemblerError> for DetectorFrameAssemblerProcessError {
    fn from(error: DetectorFrameAssemblerError) -> Self {
        Self::Assembly(error)
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 detector readout block to complete-frame assembly.
    pub DetectorFrameAssemblerF32 {
        label = "detector-frame-assembler-f32";
        configuration DetectorFrameAssemblerF32Configuration {
            detector_size: (usize, usize) = GraphRebuild;
            region_lines: usize = GraphRebuild;
            block_lines: usize = GraphRebuild;
            region_samples: usize = GraphRebuild;
            detector_origins: Vec<[usize; 2]> = GraphRebuild;
            detector_line_steps: Vec<[i32; 2]> = GraphRebuild;
            detector_sample_steps: Vec<[i32; 2]> = GraphRebuild;
        }
        plan = DetectorFrameAssemblerPlan;
        workspace = DetectorFrameAssemblerWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            detector_rows = plan.readout().detector_size().0;
            detector_columns = plan.readout().detector_size().1;
            readout_region_count = plan.readout().regions().len();
            block_lines = plan.readout().block_lines();
            region_samples = plan.readout().region_samples();
            block_rate_multiplier = u32::try_from(plan.readout().blocks())
                .map_err(|_| crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            readout_block: f32 [readout_region_count, block_lines, region_samples]
                = "readout-block",
                schema = crate::schemas::CALIBRATED_PIXEL_READOUT_BLOCK_V1,
                sample = readout_sample,
                rate = [block_rate_multiplier, 1];
        }
        outputs {
            detector_frame: f32 [detector_rows, detector_columns] = "detector-frame",
                schema = crate::schemas::CALIBRATED_PIXELS_V1,
                metadata_from = "readout-block",
                publication = conditional;
        }
        prepare |configuration| {
            let readout = DetectorReadoutPlan::from_i32_coordinate_steps(
                configuration.detector_size,
                configuration.region_lines,
                configuration.block_lines,
                configuration.region_samples,
                &configuration.detector_origins,
                &configuration.detector_line_steps,
                &configuration.detector_sample_steps,
            )?;
            let plan = DetectorFrameAssemblerPlan::new(readout);
            let workspace = plan.workspace::<f32>();
            Ok((plan, workspace))
        }
        prepare_error = DetectorReadoutProfileError;
        process |plan, workspace| {
            let Some(sample) = readout_sample else {
                workspace.reset();
                return Err(DetectorFrameAssemblerProcessError::MissingSampleMetadata);
            };
            let ordinal = usize::try_from(sample.offset())
                .map_err(|_| DetectorFrameAssemblerProcessError::OrdinalOverflow)?;
            let result = plan.process_block(
                detector_frame,
                workspace,
                readout_block,
                DetectorReadoutBlockInfo {
                    sequence: sample.sequence(),
                    ordinal,
                    terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(),
                    corrupted: sample.is_corrupted(),
                },
            )?;
            Ok(match result {
                DetectorReadoutResult::Pending | DetectorReadoutResult::Dropped(_) => {
                    crate::AlgorithmProcessResult::deferred()
                }
                DetectorReadoutResult::Complete {
                    sequence: _,
                    discontinuity,
                } => crate::AlgorithmProcessResult::complete_with_metadata(
                    sample.with_offset(0)

                        .with_terminal(true)
                        .with_discontinuity(discontinuity).with_corrupted(false),
                ),
            })
        }
        process_error = DetectorFrameAssemblerProcessError;
        reset |_plan, workspace| {
            workspace.reset();
        }
        plan_adopted |_plan, workspace| {
            workspace.reset();
        }
    }
}
