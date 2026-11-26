//! Exact raw detector ADU sums over the four configured pyramid rectangles.
use crate::{AlgorithmPlan, NoWorkspace};
use alloc::vec::Vec;
use core::fmt;

/// Invalid four-pupil geometry or changed caller buffer extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPupilFluxError {
    /// Geometry is empty, outside the detector, or exceeds native port bounds.
    Geometry,
    /// The maximum raw UInt16 sum exceeds the exact Float64 integer range.
    ExactSum,
    /// A caller buffer does not match its prepared extent.
    PortExtent,
}
impl fmt::Display for PyramidPupilFluxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Geometry => "pyramid pupil flux geometry is invalid",
            Self::ExactSum => "pyramid pupil flux can exceed the exact Float64 integer range",
            Self::PortExtent => "pyramid pupil flux port extent changed",
        })
    }
}
impl core::error::Error for PyramidPupilFluxError {}

/// Immutable zero-based detector/pupil geometry. No calibration or history.
#[derive(Clone, Debug)]
pub struct PyramidPupilFluxPlan {
    image_size: (usize, usize),
    pupil_size: (usize, usize),
    origins: [[usize; 2]; 4],
    image_extent: usize,
}
impl PyramidPupilFluxPlan {
    /// Prepare four rectangles in configured origin order, including overlaps.
    pub fn new(
        image_size: (usize, usize),
        pupil_size: (usize, usize),
        origins: Vec<[usize; 2]>,
    ) -> Result<Self, PyramidPupilFluxError> {
        use PyramidPupilFluxError as E;
        let [rows, columns] = [image_size.0, image_size.1];
        let [pupil_rows, pupil_columns] = [pupil_size.0, pupil_size.1];
        if rows == 0 || columns == 0 || pupil_rows == 0 || pupil_columns == 0 {
            return Err(E::Geometry);
        }
        let pixels = pupil_rows.checked_mul(pupil_columns).ok_or(E::ExactSum)?;
        let maximum = (pixels as u128) * u128::from(u16::MAX);
        if maximum > 1_u128 << 53 {
            return Err(E::ExactSum);
        }
        let image_extent = rows
            .checked_mul(columns)
            .filter(|&n| n <= u32::MAX as usize / 2)
            .ok_or(E::Geometry)?;
        let origins: [[usize; 2]; 4] = origins.try_into().map_err(|_| E::Geometry)?;
        for [row, column] in origins {
            if row.checked_add(pupil_rows).is_none_or(|n| n > rows)
                || column
                    .checked_add(pupil_columns)
                    .is_none_or(|n| n > columns)
            {
                return Err(E::Geometry);
            }
        }
        Ok(Self {
            image_size,
            pupil_size,
            origins,
            image_extent,
        })
    }
}
impl AlgorithmPlan for PyramidPupilFluxPlan {
    type Input<'a> = &'a [u16];
    type Output<'a> = &'a mut [f64];
    type Workspace = NoWorkspace;
    type Error = PyramidPupilFluxError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _: &mut NoWorkspace,
        raw: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if raw.len() != self.image_extent || output.len() != 4 {
            return Err(PyramidPupilFluxError::PortExtent);
        }
        for (flux, [first_row, first_column]) in output.iter_mut().zip(self.origins) {
            let mut total = 0.0;
            for row in first_row..first_row + self.pupil_size.0 {
                for column in first_column..first_column + self.pupil_size.1 {
                    total += f64::from(raw[row * self.image_size.1 + column]);
                }
            }
            *flux = total;
        }
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Four exact Float64 raw detector ADU sums in configured pupil order.
    pub PyramidPupilFluxU16F64 {
        label="pyramid-pupil-flux-u16-f64";
        configuration PyramidPupilFluxU16F64Configuration {
            image_size:(usize,usize)=GraphRebuild;
            pupil_size:(usize,usize)=GraphRebuild;
            pupil_origins:Vec<[usize;2]>=GraphRebuild;
        }
        plan=PyramidPupilFluxPlan;
        properties=none;
        dimensions |plan,_workspace| {rows=plan.image_size.0;columns=plan.image_size.1;}
        inputs {raw:u16[rows,columns]="raw",schema=crate::schemas::RAW_DETECTOR_PIXELS_V1;}
        outputs {flux:f64[4]="pupil-flux",schema=crate::schemas::PYRAMID_PUPIL_FLUX_ADU_F64_V1,metadata_from="raw";}
        prepare |configuration| {
            let plan=PyramidPupilFluxPlan::new(configuration.image_size,configuration.pupil_size,configuration.pupil_origins)?;
            Ok((plan,NoWorkspace))
        }
        prepare_error=PyramidPupilFluxError;
        process |plan,workspace| {plan.process(flux,workspace,raw)}
        process_error=PyramidPupilFluxError;
        reset |_plan,_workspace| {}
    }
}
