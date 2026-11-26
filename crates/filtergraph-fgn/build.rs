//! Narrows generated Linux example libraries to the admitted FGN ABI surface.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=fgn-exports.map");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let exports = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo supplies CARGO_MANIFEST_DIR"),
    )
    .join("fgn-exports.map");
    println!(
        "cargo:rustc-link-arg-examples=-Wl,--version-script={}",
        exports.display()
    );
}
