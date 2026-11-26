//! Typed, ABI-free port access for safe FilterGraph deployment declarations.

use core::marker::PhantomData;

use crate::{InputBuffers, NdarrayElement, OutputBuffers, ParameterBuffer};

/// Typed access to one per-frame ndarray input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataInput<T> {
    input: usize,
    element: PhantomData<fn() -> T>,
}

impl<T> DataInput<T> {
    #[doc(hidden)]
    pub const fn new(input: usize) -> Self {
        Self {
            input,
            element: PhantomData,
        }
    }

    /// Returns the direction-local process input index.
    pub const fn index(self) -> usize {
        self.input
    }
}

impl<T> DataInput<T>
where
    T: NdarrayElement,
{
    /// Borrows the complete packed input after checking its element type and
    /// extent.
    pub fn slice<'a>(self, inputs: &InputBuffers<'a>, elements: usize) -> Result<&'a [T], i32> {
        inputs.slice(self.input, elements)
    }
}

/// Typed access to one sparse ndarray parameter input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParameterInput<T> {
    port: u32,
    input: usize,
    element: PhantomData<fn() -> T>,
}

impl<T> ParameterInput<T> {
    #[doc(hidden)]
    pub const fn new(port: u32, input: usize) -> Self {
        Self {
            port,
            input,
            element: PhantomData,
        }
    }

    /// Returns the descriptor-local port identifier used by sparse updates.
    pub const fn port(self) -> u32 {
        self.port
    }

    /// Returns the direction-local process input index.
    pub const fn input_index(self) -> usize {
        self.input
    }

    /// Verifies that a process call did not attach a per-frame buffer to this
    /// sparse parameter port.
    pub fn is_disconnected(self, inputs: &InputBuffers<'_>) -> Result<bool, i32> {
        inputs.is_disconnected(self.input)
    }
}

impl<T> ParameterInput<T>
where
    T: NdarrayElement,
{
    /// Borrows a checked packed parameter value during off-loop preparation.
    pub fn slice<'a>(
        self,
        parameter: ParameterBuffer<'a>,
        elements: usize,
    ) -> Result<&'a [T], i32> {
        parameter.slice(elements)
    }
}

/// Typed access to one per-frame ndarray output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataOutput<T> {
    output: usize,
    element: PhantomData<fn() -> T>,
}

impl<T> DataOutput<T> {
    #[doc(hidden)]
    pub const fn new(output: usize) -> Self {
        Self {
            output,
            element: PhantomData,
        }
    }

    /// Returns the direction-local process output index.
    pub const fn index(self) -> usize {
        self.output
    }

    /// Copies matching metadata from one data input.
    pub fn copy_metadata_from<U>(
        self,
        outputs: &mut OutputBuffers<'_>,
        inputs: &InputBuffers<'_>,
        source: DataInput<U>,
    ) -> Result<(), i32> {
        outputs.copy_metadata_from(self.output, inputs, source.input)
    }
}

impl<T> DataOutput<T>
where
    T: NdarrayElement,
{
    /// Borrows writable packed output storage after checking its element type
    /// and extent.
    pub fn slice_mut<'a>(
        self,
        outputs: &'a mut OutputBuffers<'_>,
        elements: usize,
    ) -> Result<&'a mut [T], i32> {
        outputs.slice_mut(self.output, elements)
    }

    /// Copies a complete staged result into checked output storage.
    pub fn copy_from(self, outputs: &mut OutputBuffers<'_>, values: &[T]) -> Result<(), i32> {
        self.slice_mut(outputs, values.len())?
            .copy_from_slice(values);
        Ok(())
    }

    /// Publishes the completed output byte count after successful processing.
    pub fn finish(self, outputs: &mut OutputBuffers<'_>, elements: usize) -> Result<(), i32> {
        outputs.finish::<T>(self.output, elements)
    }
}

/// Declare one statically shaped port topology and generate typed process
/// handles.
///
/// The declaration groups per-frame inputs, outputs, and sparse parameter
/// inputs. It generates the FGN port table, descriptor identifiers, direction-
/// local process indices, C strings, and element-typed buffer accessors. Safe
/// deployment code therefore names scientific ports without constructing ABI
/// structs or repeating magic indices.
#[macro_export]
macro_rules! declare_ndarray_ports {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident {
            inputs {
                $($input:ident: $input_type:ty = $input_name:literal;)*
            }
            outputs {
                $($output:ident: $output_type:ty = $output_name:literal;)*
            }
            parameters {
                $($parameter:ident: $parameter_type:ty = $parameter_name:literal;)*
            }
        }
    ) => {
        $(#[$meta])*
        $vis struct $name;

        impl $name {
            /// Static adapter-owned FGN port table.
            pub const PORTS: &'static [$crate::abi::PortInfo] = &$crate::declare_ndarray_ports!(
                @port_array [0_u32] [];
                $(input $input_name;)*
                $(output $output_name;)*
                $(parameter $parameter_name;)*
            );

            $crate::declare_ndarray_ports!(@input_constants [0_usize];
                $($input: $input_type = $input_name;)*
            );
            $crate::declare_ndarray_ports!(@output_constants [0_usize];
                $($output: $output_type = $output_name;)*
            );
            $crate::declare_ndarray_ports!(@parameter_constants
                [0_usize $(+ { let _ = stringify!($input); 1_usize })*]
                [0_u32
                    $(+ { let _ = stringify!($input); 1_u32 })*
                    $(+ { let _ = stringify!($output); 1_u32 })*
                ];
                $($parameter: $parameter_type = $parameter_name;)*
            );
        }
    };

    (@port_array [$index:expr] [$($ports:expr,)*];) => {
        [$($ports,)*]
    };
    (@port_array [$index:expr] [$($ports:expr,)*]; input $name:literal; $($rest:tt)*) => {
        $crate::declare_ndarray_ports!(@port_array [($index + 1_u32)] [
            $($ports,)*
            $crate::abi::PortInfo::new(
                $index,
                $crate::abi::DIRECTION_INPUT,
                0,
                concat!($name, "\0").as_ptr().cast(),
            ),
        ]; $($rest)*)
    };
    (@port_array [$index:expr] [$($ports:expr,)*]; output $name:literal; $($rest:tt)*) => {
        $crate::declare_ndarray_ports!(@port_array [($index + 1_u32)] [
            $($ports,)*
            $crate::abi::PortInfo::new(
                $index,
                $crate::abi::DIRECTION_OUTPUT,
                0,
                concat!($name, "\0").as_ptr().cast(),
            ),
        ]; $($rest)*)
    };
    (@port_array [$index:expr] [$($ports:expr,)*]; parameter $name:literal; $($rest:tt)*) => {
        $crate::declare_ndarray_ports!(@port_array [($index + 1_u32)] [
            $($ports,)*
            $crate::abi::PortInfo::new(
                $index,
                $crate::abi::DIRECTION_INPUT,
                $crate::abi::PORT_OPTIONAL | $crate::abi::PORT_PARAMETER,
                concat!($name, "\0").as_ptr().cast(),
            ),
        ]; $($rest)*)
    };

    (@input_constants [$index:expr];) => {};
    (@input_constants [$index:expr];
        $port:ident: $element:ty = $name:literal; $($rest:tt)*
    ) => {
        #[doc = concat!("Typed per-frame input handle for `", $name, "`.")]
        pub const $port: $crate::DataInput<$element> = $crate::DataInput::new($index);
        $crate::declare_ndarray_ports!(@input_constants [($index + 1_usize)]; $($rest)*);
    };

    (@output_constants [$index:expr];) => {};
    (@output_constants [$index:expr];
        $port:ident: $element:ty = $name:literal; $($rest:tt)*
    ) => {
        #[doc = concat!("Typed per-frame output handle for `", $name, "`.")]
        pub const $port: $crate::DataOutput<$element> = $crate::DataOutput::new($index);
        $crate::declare_ndarray_ports!(@output_constants [($index + 1_usize)]; $($rest)*);
    };

    (@parameter_constants [$input_index:expr] [$port:expr];) => {};
    (@parameter_constants [$input_index:expr] [$port:expr];
        $name:ident: $element:ty = $label:literal; $($rest:tt)*
    ) => {
        #[doc = concat!("Typed sparse parameter handle for `", $label, "`.")]
        pub const $name: $crate::ParameterInput<$element> =
            $crate::ParameterInput::new($port, $input_index);
        $crate::declare_ndarray_ports!(@parameter_constants
            [($input_index + 1_usize)] [($port + 1_u32)]; $($rest)*
        );
    };
}
