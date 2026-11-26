//! Keep the production shared library's dynamic surface at the FGN entry point.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn git(directory: &std::path::Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../..");
    // An unpacked source archive may be inside an unrelated parent Git checkout.
    let tracked = git(
        &root,
        &[
            "ls-files",
            "--error-unmatch",
            "crates/filtergraph-fgn-bundle/src/lib.rs",
        ],
    )
    .is_some();
    let commit = tracked
        .then(|| git(&root, &["rev-parse", "HEAD"]))
        .flatten()
        .map(|commit| format!("\"{commit}\""))
        .unwrap_or_else(|| "null".into());
    let dirty = tracked
        .then(|| {
            git(
                &root,
                &["status", "--porcelain", "--untracked-files=normal"],
            )
        })
        .flatten()
        .map(|status| if status.is_empty() { "false" } else { "true" })
        .unwrap_or("null");
    println!("cargo:rustc-env=FILTERGRAPH_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FILTERGRAPH_GIT_DIRTY={dirty}");
    // Rebuild the stamp when source or Git metadata changes, including worktrees.
    for args in [
        ["rev-parse", "--git-path", "HEAD"],
        ["rev-parse", "--git-path", "index"],
    ] {
        if let Some(path) = git(&root, &args) {
            println!("cargo:rerun-if-changed={}", root.join(path).display());
        }
    }
    if let Some(reference) = git(&root, &["symbolic-ref", "-q", "HEAD"])
        && let Some(path) = git(&root, &["rev-parse", "--git-path", &reference])
    {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    if let Some(files) = git(
        &root,
        &["ls-files", "--cached", "--others", "--exclude-standard"],
    ) {
        for file in files.lines() {
            println!("cargo:rerun-if-changed={}", root.join(file).display());
        }
    }
    println!("cargo:rerun-if-changed=fgn-exports.map");

    if !matches!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("linux" | "android" | "freebsd")
    ) {
        return;
    }

    let exports = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo supplies CARGO_MANIFEST_DIR"),
    )
    .join("fgn-exports.map");
    println!("cargo:rustc-cdylib-link-arg=-Wl,--exclude-libs,ALL");
    println!(
        "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
        exports.display()
    );
}
