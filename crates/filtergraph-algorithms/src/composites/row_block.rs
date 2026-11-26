//! Shared scientific identity and recovery for complete detector-row blocks.

use core::mem;

/// Portable identity and placement of one complete detector-row block.
///
/// This is scientific sample information rather than a transport header. An
/// FGN adapter maps standard SPA Header fields to this value; a native caller
/// or another graph runtime can construct it directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowBlockInfo {
    /// Stable detector-frame identity shared by all blocks in one frame.
    pub sequence: u64,
    /// Zero-based first detector row in this block.
    pub first_row: usize,
    /// Whether this is the final block in the detector frame.
    pub terminal: bool,
    /// Whether this frame follows a loss or abandoned frame.
    pub discontinuity: bool,
    /// Whether acquisition marked this block corrupt.
    pub corrupted: bool,
}

/// Recoverable reason that one partial detector frame was abandoned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowBlockDropReason {
    /// A non-initial block arrived without an active frame.
    MissingStart,
    /// A block used a different sequence from the active frame.
    Sequence,
    /// The next block did not begin at the next required detector row.
    RowGap,
    /// A block range exceeded the prepared detector image.
    RowRange,
    /// The terminal indication did not match the prepared detector height.
    Terminal,
    /// Discontinuity was asserted after the first block of a frame.
    MidFrameDiscontinuity,
    /// Acquisition marked the block corrupt.
    Corrupted,
}

/// Publication result after consuming one complete row block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowBlockResult {
    /// The frame remains in progress and caller-owned output is unchanged.
    Pending,
    /// A complete frame-rate result was written to caller-owned output.
    Complete {
        /// Stable detector-frame identity of the completed result.
        sequence: u64,
        /// Whether the result follows any reported or locally detected loss.
        discontinuity: bool,
    },
    /// The supplied block abandoned a partial frame without publishing output.
    Dropped(RowBlockDropReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AcceptedRowBlock {
    pub(crate) end_row: usize,
}

/// Common single-owner ordering state for fixed-height detector-row blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RowBlockFrame {
    sequence: Option<u64>,
    expected_row: usize,
    pending_discontinuity: bool,
    current_discontinuity: bool,
}

impl RowBlockFrame {
    pub(crate) const fn new() -> Self {
        Self {
            sequence: None,
            expected_row: 0,
            pending_discontinuity: false,
            current_discontinuity: false,
        }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::new();
    }

    pub(crate) const fn sequence(&self) -> Option<u64> {
        self.sequence
    }

    pub(crate) const fn discontinuity(&self) -> bool {
        self.current_discontinuity
    }

    fn start(&mut self, info: RowBlockInfo) {
        self.sequence = Some(info.sequence);
        self.expected_row = 0;
        self.current_discontinuity =
            mem::take(&mut self.pending_discontinuity) || info.discontinuity;
    }

    pub(crate) fn accept(
        &mut self,
        info: RowBlockInfo,
        block_rows: usize,
        image_rows: usize,
    ) -> Result<AcceptedRowBlock, RowBlockDropReason> {
        if info.corrupted {
            self.abandon();
            return Err(RowBlockDropReason::Corrupted);
        }
        if info.first_row == 0 {
            let superseded = self.sequence.is_some();
            if superseded {
                self.abandon();
            }
            self.start(info);
            if superseded {
                self.current_discontinuity = true;
            }
        } else {
            let Some(sequence) = self.sequence else {
                self.abandon();
                return Err(RowBlockDropReason::MissingStart);
            };
            if sequence != info.sequence {
                self.abandon();
                return Err(RowBlockDropReason::Sequence);
            }
            if info.discontinuity {
                self.abandon();
                return Err(RowBlockDropReason::MidFrameDiscontinuity);
            }
        }
        if info.first_row != self.expected_row {
            self.abandon();
            return Err(RowBlockDropReason::RowGap);
        }
        let Some(end_row) = info.first_row.checked_add(block_rows) else {
            self.abandon();
            return Err(RowBlockDropReason::RowRange);
        };
        if end_row > image_rows {
            self.abandon();
            return Err(RowBlockDropReason::RowRange);
        }
        if info.terminal != (end_row == image_rows) {
            self.abandon();
            return Err(RowBlockDropReason::Terminal);
        }
        self.expected_row = end_row;
        Ok(AcceptedRowBlock { end_row })
    }

    pub(crate) fn abandon(&mut self) {
        self.sequence = None;
        self.expected_row = 0;
        self.pending_discontinuity = true;
        self.current_discontinuity = false;
    }

    pub(crate) fn complete(&mut self) {
        self.sequence = None;
        self.expected_row = 0;
        self.pending_discontinuity = false;
        self.current_discontinuity = false;
    }
}
