//! Emit fixed stepped-topology leaky-integrator reference outputs.
//!
//! The input sequences and transitions match the RTC live topology fixture.
//! Frames 1 through 6 use `gain = 0.5`, `pole = 0.75`; frames 5 and 6 retain
//! state across finite-source completion and restart. Before frame 7, every
//! stage resets, then runs frames 7 and 8 with `gain = 0.25`, `pole = 0.75`.

use filtergraph_algorithms::{AlgorithmPlan, LeakyIntegratorPlan, LeakyIntegratorWorkspace};

const INPUTS_A: [[f32; 2]; 8] = [
    [1.0, -1.0],
    [0.25, 2.5],
    [-3.0, 0.125],
    [4.5, -2.25],
    [1.5, -0.75],
    [-1.25, 3.25],
    [2.0, -4.0],
    [0.375, 1.125],
];
const INPUTS_B: [[f32; 2]; 8] = [
    [-0.5, 3.0],
    [2.25, -1.5],
    [0.75, 4.0],
    [-2.0, 0.5],
    [3.5, -2.0],
    [-0.125, 1.75],
    [4.25, -3.0],
    [-2.5, 0.625],
];
const FNV1A_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV1A_PRIME: u64 = 1_099_511_628_211;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Topology {
    Direct,
    Serial,
    Fork,
    Independent,
}

impl Topology {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "direct" => Ok(Self::Direct),
            "serial" => Ok(Self::Serial),
            "fork" => Ok(Self::Fork),
            "independent" => Ok(Self::Independent),
            _ => Err(format!("unknown topology {value:?}")),
        }
    }

    const fn is_serial(self) -> bool {
        matches!(self, Self::Serial)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Format {
    Bits,
    Digest,
}

impl Format {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bits" => Ok(Self::Bits),
            "digest" => Ok(Self::Digest),
            _ => Err(format!("unknown format {value:?}")),
        }
    }
}

fn inputs(variant: &str) -> Result<&'static [[f32; 2]; 8], String> {
    match variant {
        "a" => Ok(&INPUTS_A),
        "b" => Ok(&INPUTS_B),
        _ => Err(format!("unknown input variant {variant:?}")),
    }
}

fn run(topology: Topology, inputs: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut plan = LeakyIntegratorPlan::new(0.5_f32, 0.75).expect("fixed coefficients are valid");
    let mut first = LeakyIntegratorWorkspace::new(2).expect("fixed extent is positive");
    let mut second = LeakyIntegratorWorkspace::new(2).expect("fixed extent is positive");
    let mut first_output = [0.0_f32; 2];
    let mut output = [0.0_f32; 2];
    let mut outputs = Vec::with_capacity(inputs.len());

    for (frame, input) in inputs.iter().enumerate() {
        if frame == 6 {
            first.reset();
            second.reset();
            plan = LeakyIntegratorPlan::new(0.25_f32, 0.75).expect("fixed coefficients are valid");
        }
        plan.process(&mut first_output, &mut first, input)
            .expect("fixed extent matches workspace");
        if topology.is_serial() {
            plan.process(&mut output, &mut second, &first_output)
                .expect("fixed extent matches workspace");
        } else {
            output = first_output;
        }
        outputs.push(output);
    }
    outputs
}

fn prefix_digests(outputs: &[[f32; 2]]) -> Vec<u64> {
    let mut digest = FNV1A_OFFSET_BASIS;
    let mut digests = Vec::with_capacity(outputs.len());
    for output in outputs {
        for value in output {
            for byte in value.to_le_bytes() {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(FNV1A_PRIME);
            }
        }
        digests.push(digest);
    }
    digests
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} <direct|serial|fork|independent> <a|b> <bits|digest>\n\
         Emits one ordered line per stepped input. `bits` emits exact f32 bit patterns; \
         `digest` emits the cumulative FNV-1a 64-bit payload digest."
    )
}

fn main() {
    let mut arguments = std::env::args();
    let program = arguments
        .next()
        .unwrap_or_else(|| String::from("leaky_integrator_oracle"));
    let result = (|| {
        let topology = Topology::parse(&arguments.next().ok_or_else(|| usage(&program))?)?;
        let input_variant = arguments.next().ok_or_else(|| usage(&program))?;
        let format = Format::parse(&arguments.next().ok_or_else(|| usage(&program))?)?;
        if arguments.next().is_some() {
            return Err(usage(&program));
        }

        let outputs = run(topology, inputs(&input_variant)?);
        match format {
            Format::Bits => {
                for (step, output) in outputs.iter().enumerate() {
                    println!(
                        "{} {:08x} {:08x}",
                        step + 1,
                        output[0].to_bits(),
                        output[1].to_bits()
                    );
                }
            }
            Format::Digest => {
                for (step, digest) in prefix_digests(&outputs).iter().enumerate() {
                    println!("{} {digest:016x}", step + 1);
                }
            }
        }
        Ok(())
    })();

    if let Err(message) = result {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_topology_transitions_use_the_maintained_plan() {
        assert_eq!(
            run(Topology::Direct, &INPUTS_A)
                .iter()
                .map(|output| [output[0].to_bits(), output[1].to_bits()])
                .collect::<Vec<_>>(),
            vec![
                [0x3f00_0000, 0xbf00_0000],
                [0x3f00_0000, 0x3f60_0000],
                [0xbf90_0000, 0x3f38_0000],
                [0x3fb4_0000, 0xbf16_0000],
                [0x3fe7_0000, 0xbf50_8000],
                [0x3f3a_8000, 0x3f81_d000],
                [0x3f00_0000, 0xbf80_0000],
                [0x3ef0_0000, 0xbef0_0000],
            ]
        );
        assert_eq!(
            run(Topology::Serial, &INPUTS_A)
                .iter()
                .map(|output| [output[0].to_bits(), output[1].to_bits()])
                .collect::<Vec<_>>(),
            vec![
                [0x3e80_0000, 0xbe80_0000],
                [0x3ee0_0000, 0x3e80_0000],
                [0xbe70_0000, 0x3f0c_0000],
                [0x3f07_0000, 0x3df0_0000],
                [0x3fa6_2000, 0xbea3_8000],
                [0x3fab_3800, 0x3e89_0000],
                [0x3e00_0000, 0xbe80_0000],
                [0x3e58_0000, 0xbe9c_0000],
            ]
        );
        assert_eq!(
            prefix_digests(&run(Topology::Independent, &INPUTS_B)),
            vec![
                0x6cc1_2f57_51ee_7cd2,
                0x8a20_bb8d_2561_bbe5,
                0xa70a_f8c5_d364_230c,
                0xddac_204d_5d8f_66ec,
                0x5f71_fee4_7bfc_49a3,
                0x886a_6948_9555_3dfc,
                0xa06d_9acc_3d8b_71a4,
                0x4497_370d_4a3a_c004,
            ]
        );
    }
}
