//! Opt-in acquisition admission follows physical identity independently of publication.
use core::num::NonZeroU64;
use filtergraph_algorithms::{
    ChoppedImageDifferenceError as Error, ChoppedImageDifferenceF32 as D,
    ChoppedImageDifferenceF32Configuration as Configuration, ChoppedImageKind as Kind,
    DeclarationProcessError, NativeProcessError, PreparedAlgorithm, SampleAcquisitionIdentity,
    SampleMetadata,
};

fn prepared(required: bool) -> PreparedAlgorithm<D> {
    D::prepare(Configuration {
        pixel_count: 1,
        trigger_high_is_fringed: true,
        input_schema: "test.image/1".into(),
        require_acquisition_identity: required,
    })
    .unwrap()
}
fn sample(sequence: u64, kind: Kind) -> SampleMetadata {
    SampleMetadata::new(1000 - sequence, 0)
        .with_acquisition_identity(Some(
            SampleAcquisitionIdentity::new([1; 16], 3, sequence).unwrap(),
        ))
        .with_exposure_duration_nanoseconds(NonZeroU64::new(20))
        .with_chopped_image_kind(Some(kind))
}
fn process(p: &mut PreparedAlgorithm<D>, sample: SampleMetadata) -> Result<(f32, bool, u8), Error> {
    let mut difference = [123.0];
    let mut validity = [true];
    let mut status = [99];
    let result = D::process_with_metadata(
        p,
        &[Some(sample), None],
        &[5.0],
        &[true],
        &mut difference,
        &mut validity,
        &mut status,
    );
    match result {
        Ok(_) => Ok((difference[0], validity[0], status[0])),
        Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(error))) => {
            assert_eq!((difference, validity, status), ([123.0], [true], [99]));
            Err(error)
        }
        Err(error) => panic!("unexpected boundary error: {error:?}"),
    }
}

#[test]
fn required_facts_reject_without_advancing_acquisition_or_outputs() {
    let mut p = prepared(true);
    assert_eq!(
        p.required_sample_metadata(0),
        &[
            "acquisition_identity",
            "exposure_duration_nanoseconds",
            "chopped_image_kind"
        ]
    );
    assert!(p.required_sample_metadata(1).is_empty());
    let first = sample(10, Kind::Fringed);
    for (invalid, error) in [
        (
            first.with_acquisition_identity(None),
            Error::MetadataMissing,
        ),
        (
            first.with_exposure_duration_nanoseconds(None),
            Error::MetadataMissing,
        ),
        (first.with_chopped_image_kind(None), Error::TriggerMissing),
    ] {
        assert_eq!(process(&mut p, invalid), Err(error));
    }
    assert_eq!(process(&mut p, first), Ok((0.0, false, 0)));
    let next = sample(11, Kind::Unfringed);
    assert_eq!(
        process(
            &mut p,
            next.with_exposure_duration_nanoseconds(NonZeroU64::new(21))
        ),
        Err(Error::ExposureMismatch)
    );
    assert_eq!(process(&mut p, next), Ok((0.0, true, 1)));
    assert_eq!(process(&mut p, next), Err(Error::SequenceInvalid));
    let backwards = next.with_acquisition_identity(Some(
        SampleAcquisitionIdentity::new([1; 16], 2, 12).unwrap(),
    ));
    assert_eq!(process(&mut p, backwards), Err(Error::SequenceInvalid));
    assert_eq!(
        process(&mut p, sample(12, Kind::Fringed)),
        Ok((0.0, true, 1))
    );
}

#[test]
fn gaps_cohorts_discontinuity_mixed_and_corruption_reprime() {
    let mut p = prepared(true);
    assert_eq!(
        process(&mut p, sample(10, Kind::Fringed)),
        Ok((0.0, false, 0))
    );
    assert_eq!(
        process(&mut p, sample(12, Kind::Unfringed)),
        Ok((0.0, false, 0))
    );
    assert_eq!(
        process(&mut p, sample(13, Kind::Fringed)),
        Ok((0.0, true, 1))
    );
    let new_domain = sample(1, Kind::Unfringed)
        .with_acquisition_identity(Some(SampleAcquisitionIdentity::new([2; 16], 1, 1).unwrap()));
    assert_eq!(process(&mut p, new_domain), Ok((0.0, false, 0)));
    let bump = sample(14, Kind::Fringed).with_acquisition_identity(Some(
        SampleAcquisitionIdentity::new([2; 16], 2, 14).unwrap(),
    ));
    assert_eq!(process(&mut p, bump), Ok((0.0, false, 0)));
    let next = |sequence, kind| {
        sample(sequence, kind).with_acquisition_identity(Some(
            SampleAcquisitionIdentity::new([2; 16], 2, sequence).unwrap(),
        ))
    };
    assert_eq!(
        process(&mut p, next(15, Kind::Unfringed).with_discontinuity(true)),
        Ok((0.0, false, 0))
    );
    assert_eq!(process(&mut p, next(16, Kind::Fringed)), Ok((0.0, true, 1)));
    assert_eq!(process(&mut p, next(17, Kind::Mixed)), Ok((0.0, false, 0)));
    assert_eq!(
        process(&mut p, next(18, Kind::Unfringed)),
        Ok((0.0, false, 0))
    );
    assert_eq!(process(&mut p, next(19, Kind::Fringed)), Ok((0.0, true, 1)));
    assert_eq!(
        process(&mut p, next(20, Kind::Unfringed).with_corrupted(true)),
        Ok((0.0, false, 0))
    );
    assert_eq!(
        process(&mut p, next(21, Kind::Fringed)),
        Ok((0.0, false, 0))
    );
    assert_eq!(
        process(&mut p, next(22, Kind::Unfringed)),
        Ok((0.0, true, 1))
    );
    D::reset(&mut p);
    assert_eq!(
        process(&mut p, sample(1, Kind::Fringed)),
        Ok((0.0, false, 0))
    );
}

#[test]
fn default_mode_allows_unqualified_samples() {
    let mut p = prepared(false);
    assert!(p.required_sample_metadata(0).is_empty());
    assert_eq!(
        process(
            &mut p,
            SampleMetadata::new(90, 0).with_chopped_image_kind(Some(Kind::Fringed))
        ),
        Ok((0.0, false, 0))
    );
    assert_eq!(
        process(
            &mut p,
            SampleMetadata::new(2, 0).with_chopped_image_kind(Some(Kind::Unfringed))
        ),
        Ok((0.0, true, 1))
    );
}
