//! Transport-independent numerical algorithms and execution contracts for
//! FilterGraph.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod algorithms;
pub mod composites;
mod declaration;
mod interface;
pub mod schemas;

pub use algorithms::*;
pub use composites::*;
pub use declaration::{
    AlgorithmConfiguration, AlgorithmDeclaration, AlgorithmFormat, AlgorithmInput, AlgorithmInputs,
    AlgorithmOperation, AlgorithmOperationPlan, AlgorithmOperationWorkspace, AlgorithmOutput,
    AlgorithmOutputs, AlgorithmPort, AlgorithmProcessResult, ArrayLayout, ConfigurationUpdate,
    ConfigurationValue, DeclarationParameterError, DeclarationParameterPolicy,
    DeclarationPreparationError, DeclarationProcessError, DeclarationPropertyPolicy,
    DeclaredParameters, ElementType, FormatError, IntoAlgorithmProcessResult, NativeProcessError,
    NdarrayElement, NoDeclaredParameters, NoDeclaredProperties, PortDirection, PortError,
    PortPublication, PortRole, PreparedAlgorithm, PropertyPolicy, RuntimeDeclaredProperties,
    SampleAcquisitionIdentity, SampleAcquisitionIdentityError, SampleClock, SampleExposureStart,
    SampleMetadata, SampleReferenceTimestamp, SampleReferenceTimestampError, SampleTriggerState,
    ScalarOrVector, ScalarOrVectorExtentError,
};
pub use interface::{
    AlgorithmPlan, AlgorithmProperties, InputProgress, NoWorkspace, OutputProgress, ProcessError,
    ProgressResult, ProgressWorkspace, PropertyAssignment, PropertyBinding, PropertyBindingError,
    PropertyChoice, PropertyConstraint, PropertyError, PropertyInfo, PropertySchemaError,
    PropertyType, PropertyUpdate, PropertyValue, property_value_satisfies, property_values_equal,
    validate_property_info, validate_property_schema,
};

/// Dependencies used by exported declarative macros.
#[doc(hidden)]
pub mod __private {
    pub use crate::declaration::ConfigurationField;
    pub use alloc::boxed::Box;
    pub use core::convert::Infallible;
    pub use serde::de::Error as DeserializeError;
    pub use serde::{Deserialize, Deserializer};

    #[inline(always)]
    pub fn invoke<R>(function: impl FnOnce() -> R) -> R {
        function()
    }
}
