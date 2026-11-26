//! Fail closed if an external build entry point drops the workspace backend policy.
fn main() {
    for (name, expected) in [
        ("INTERFACE64", "0"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("OPENBLAS_USE_LOCKING", "1"),
        ("OPENBLAS_DYNAMIC_ARCH", "1"),
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(expected),
            "filtergraph-blas requires {name}={expected}; retain the workspace .cargo/config.toml backend policy"
        );
    }

    let source = std::path::PathBuf::from(
        std::env::var_os("DEP_OPENBLAS_INCLUDE").expect("OpenBLAS source/header metadata"),
    );
    println!("cargo:rerun-if-changed=src/gemv.c");
    for (suffix, double) in [("sgemv", false), ("dgemv", true)] {
        let mut compiler = cc::Build::new();
        compiler
            .file("src/gemv.c")
            .include(&source)
            .include(source.join("interface"))
            .define("CBLAS", None)
            .define("FILTERGRAPH_FLOAT_BYTES", if double { "8" } else { "4" })
            .define("DYNAMIC_ARCH", None)
            .define("NO_WARMUP", None)
            .define("NO_AFFINITY", None)
            .define("SMALL_MATRIX_OPT", None)
            .define("EXPRECISION", None)
            .define("BUILD_SINGLE", "1")
            .define("BUILD_DOUBLE", "1")
            .define("BUILD_COMPLEX", "1")
            .define("BUILD_COMPLEX16", "1")
            .define("MAX_CPU_NUMBER", "1")
            .define("MAX_PARALLEL_NUMBER", "1")
            .define("CNAME", format!("filtergraph_cblas_{suffix}").as_str())
            .extra_warnings(false)
            .pic(true)
            .opt_level(2)
            .flag_if_supported("-fstack-clash-protection");
        if double {
            compiler.define("DOUBLE", None);
        }
        compiler.compile(&format!("filtergraph_{suffix}"));
    }
}
