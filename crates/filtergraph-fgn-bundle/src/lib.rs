//! Production FGN deployment bundle for portable FilterGraph declarations.
//!
//! This crate contains no algorithm facts or adapter callbacks. Its type-only
//! list selects which declarations are linked into one shared library; every
//! descriptor is generated from the declaration beside the scientific code.

use filtergraph_algorithms::{
    ArrayStatisticsF32, PhysicalOffsetReferenceF32, PixelValidityF32,
    ValidityGatedClosedLoopIntegratorF32,
};

use filtergraph_algorithms::{
    BiquadCascadeF32, CenterOfGravityF32, ChoppedImageDifferenceF32, ClosedLoopCorrectionF32,
    ClosedLoopIntegratorF32, ControllerToVdmF32, CorrectorAllocationF32, DetectorFrameAssemblerF32,
    DocrimeBinaryExcitationF32, DocrimeClosedLoopInteractionMatrixF32, DocrimeCovarianceF32,
    DocrimeExcitationF32, DocrimeOpenLoopInteractionMatrixF32, FrameDelayF32,
    HadamardInteractionMatrixF32, HadamardPushPullF32, IncrementalDenseReconstructorF32,
    InterActuatorStrokeLimitF32, InteractionMatrixStabilityF32, LeakyIntegratorF32,
    MatchedCalibrationDifferencingF32, ModalPushPullF32, NdarrayCopyF32, OpticalGainCorrectionF32,
    OpticalGainDitherF32, OpticalGainEstimatorF32, OpticalGainUpdateF32, PdmCommandF32,
    PdmCommandPowerLimitF32, PdmEngineeringOffsetF32, PdmFeedbackToVdmF32,
    PdmStrokeOptimizationF32, PdmSystemFlatF32, PixelCalibrationF32, PixelCalibrationReadoutU16F32,
    PixelCalibrationRowU16F32, PixelCalibrationU16F32, PixelSelectionF32,
    PseudoOpenLoopReconstructionF32, PyramidPixelRowF32, PyramidPupilImageF32,
    PyramidReadoutReconstructorF32, PyramidReconstructorF32, PyramidRowReconstructorF32,
    RegionExtractionF32, SccPixelResidualF32, SccPushPullResponseF32, SccReconstructorF32,
    ShackHartmannImageF32, ShackHartmannMeasurementBlockF32, ShackHartmannReadoutReconstructorF32,
    ShackHartmannReconstructorF32, ShackHartmannSummaryF32, ShackHartmannWeightedCentroidF32,
    VdmFeedbackToControllerF32, VdmToPdmF32, VectorConditioningF32, VectorLinearCombinationF32,
    VectorOffsetF32, WeightedCenterOfGravityF32, ZonalInteractionMatrixF32, ZonalPushPullF32,
};

use filtergraph_algorithms::{
    BiOEdgeImageF32, CurvaturePairedChannelF32, CurvaturePairedImageF32,
    DenseVectorReconstructorF32, MlLlowfsF32, PyramidImageF32, SccCorrectionF32,
    SccSpatialCarrierDemodulationF32, SccWeightedRegularizedReconstructorF32,
    ShackHartmannReadoutMeasurementBlockF32, ShackHartmannRowReconstructorF32,
    ShackHartmannWeightedMeasurementBlockF32, VdmShapeOffsetF32, ZernikePupilSignalF32,
};

use filtergraph_algorithms::{
    PyramidMeasurementBlockF32, PyramidPupilFluxU16F64, PyramidPupilReconstructorF32,
    PyramidReadoutMeasurementBlockF32,
};

filtergraph_fgn::export_algorithm_declarations!(
    "filtergraph",
    [
        ArrayStatisticsF32,
        PhysicalOffsetReferenceF32,
        PixelValidityF32,
        ValidityGatedClosedLoopIntegratorF32,
        BiOEdgeImageF32,
        CurvaturePairedChannelF32,
        CurvaturePairedImageF32,
        DenseVectorReconstructorF32,
        MlLlowfsF32,
        PyramidImageF32,
        SccCorrectionF32,
        SccSpatialCarrierDemodulationF32,
        SccWeightedRegularizedReconstructorF32,
        ShackHartmannReadoutMeasurementBlockF32,
        ShackHartmannRowReconstructorF32,
        ShackHartmannWeightedMeasurementBlockF32,
        VdmShapeOffsetF32,
        ZernikePupilSignalF32,
        PyramidMeasurementBlockF32,
        PyramidPupilFluxU16F64,
        PyramidPupilReconstructorF32,
        PyramidReadoutMeasurementBlockF32,
        LeakyIntegratorF32,
        SccPixelResidualF32,
        SccReconstructorF32,
        ShackHartmannSummaryF32,
        ShackHartmannWeightedCentroidF32,
        ClosedLoopIntegratorF32,
        WeightedCenterOfGravityF32,
        CorrectorAllocationF32,
        PixelCalibrationF32,
        BiquadCascadeF32,
        FrameDelayF32,
        VectorConditioningF32,
        VectorLinearCombinationF32,
        VectorOffsetF32,
        DocrimeExcitationF32,
        DocrimeBinaryExcitationF32,
        PixelSelectionF32,
        NdarrayCopyF32,
        RegionExtractionF32,
        SccPushPullResponseF32,
        ChoppedImageDifferenceF32,
        CenterOfGravityF32,
        PixelCalibrationU16F32,
        PixelCalibrationRowU16F32,
        PixelCalibrationReadoutU16F32,
        DetectorFrameAssemblerF32,
        OpticalGainCorrectionF32,
        ShackHartmannImageF32,
        ShackHartmannReconstructorF32,
        ShackHartmannMeasurementBlockF32,
        IncrementalDenseReconstructorF32,
        ShackHartmannReadoutReconstructorF32,
        PdmCommandF32,
        PdmCommandPowerLimitF32,
        PdmSystemFlatF32,
        PdmEngineeringOffsetF32,
        InterActuatorStrokeLimitF32,
        PdmStrokeOptimizationF32,
        PseudoOpenLoopReconstructionF32,
        InteractionMatrixStabilityF32,
        MatchedCalibrationDifferencingF32,
        ModalPushPullF32,
        ZonalPushPullF32,
        HadamardPushPullF32,
        ZonalInteractionMatrixF32,
        HadamardInteractionMatrixF32,
        DocrimeCovarianceF32,
        DocrimeOpenLoopInteractionMatrixF32,
        DocrimeClosedLoopInteractionMatrixF32,
        OpticalGainDitherF32,
        OpticalGainEstimatorF32,
        OpticalGainUpdateF32,
        PyramidPupilImageF32,
        PyramidPixelRowF32,
        PyramidReconstructorF32,
        PyramidRowReconstructorF32,
        PyramidReadoutReconstructorF32,
        ClosedLoopCorrectionF32,
        ControllerToVdmF32,
        VdmToPdmF32,
        PdmFeedbackToVdmF32,
        VdmFeedbackToControllerF32,
    ],
);

/// Return immutable, NUL-terminated JSON describing the software at build time.
///
/// The pointer remains valid while the shared library is loaded. This optional
/// cold-path query has no FGN instance, buffer, or process-time dependency.
#[unsafe(no_mangle)]
pub extern "C" fn filtergraph_software_version() -> *const core::ffi::c_char {
    concat!(
        "{\"version\":\"",
        env!("CARGO_PKG_VERSION"),
        "\",\"git_commit\":",
        env!("FILTERGRAPH_GIT_COMMIT"),
        ",\"git_dirty\":",
        env!("FILTERGRAPH_GIT_DIRTY"),
        "}\0"
    )
    .as_ptr()
    .cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_identity_is_immutable_build_metadata() {
        let pointer = filtergraph_software_version();
        // SAFETY: the query returns a static NUL-terminated string.
        let identity = unsafe { std::ffi::CStr::from_ptr(pointer) }
            .to_str()
            .unwrap();
        assert_eq!(
            identity,
            format!(
                "{{\"version\":\"{}\",\"git_commit\":{},\"git_dirty\":{}}}",
                env!("CARGO_PKG_VERSION"),
                env!("FILTERGRAPH_GIT_COMMIT"),
                env!("FILTERGRAPH_GIT_DIRTY")
            )
        );
        assert_eq!(filtergraph_software_version(), pointer);
    }

    #[test]
    fn production_bundle_contains_every_selected_declaration() {
        assert_eq!(filtergraph_fgn_registered_declaration_count(), 84);
        let plugin =
            spa_filter_graph_ndarray_plugin_get_interface(filtergraph_fgn::abi::ABI_VERSION);
        assert!(!plugin.is_null());
        let incompatible_abi = if filtergraph_fgn::abi::ABI_VERSION == 7 {
            8
        } else {
            7
        };
        assert!(spa_filter_graph_ndarray_plugin_get_interface(incompatible_abi).is_null());
        // SAFETY: the generated registry has shared-library lifetime.
        assert_eq!(
            unsafe { (*plugin).flags },
            filtergraph_fgn::abi::PLUGIN_RETAIN_LIBRARY
        );
    }
}
