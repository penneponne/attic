//! Load the real SFNN-1536 `nn.bin` when it is present.
//!
//! The network file is staged locally and never committed, so when it is absent
//! the test prints a notice and passes and the default `cargo test` run stays
//! green everywhere.

#![cfg(feature = "sfnn")]

use std::path::PathBuf;

use attic_eval::{HIDDEN_SIZE, LAYER_STACKS, NUM_FEATURES, load_network};

/// Resolves the staged network path relative to the workspace root; the threat shape's is
/// Sirius's, under `ATTIC_SIRIUS_REF` (see `sirius_parity.rs`).
#[cfg(feature = "sfnn_threat")]
fn nn_bin_path() -> PathBuf {
    std::env::var_os("ATTIC_SIRIUS_REF")
        .map(|dir| PathBuf::from(dir).join("eval/nn.bin"))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin"))
}

#[cfg(not(feature = "sfnn_threat"))]
fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

#[test]
#[cfg_attr(miri, ignore)]
fn loads_real_network_if_present() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping loads_real_network_if_present: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }

    let net = load_network(&path).expect("real nn.bin should load and validate");

    assert_eq!(net.stacks.len(), LAYER_STACKS, "layer-stack count");
    assert_eq!(net.ft_biases.len(), HIDDEN_SIZE, "ft bias count");
    assert_eq!(
        net.ft_weights.len(),
        HIDDEN_SIZE * NUM_FEATURES,
        "ft weight count"
    );

    // Sirius's file also carries the threat block: 218,016 rows of 1,024 signed bytes.
    #[cfg(feature = "sfnn_threat")]
    {
        assert_eq!(
            net.threat_weights.len(),
            218_016 * HIDDEN_SIZE,
            "threat byte count"
        );
        assert!(
            (net.threat_weights.as_ptr() as usize).is_multiple_of(64),
            "threat block not 64-byte aligned"
        );
        assert_eq!(
            net.sha256[..4],
            [0x2d, 0x2f, 0x8e, 0x94],
            "not the Sirius nn.bin this test was written against (sha256 2d2f8e94...)"
        );
    }

    // Every buffer must land on a 64-byte boundary for the SIMD kernels.
    let is_aligned = |ptr: *const u8| (ptr as usize).is_multiple_of(64);
    assert!(is_aligned(net.ft_biases.as_ptr() as *const u8));
    assert!(is_aligned(net.ft_weights.as_ptr() as *const u8));
    for (i, stack) in net.stacks.iter().enumerate() {
        assert!(
            is_aligned(stack.fc_0_biases().as_ptr() as *const u8),
            "stack[{i}].fc_0_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_0_weights().as_ptr() as *const u8),
            "stack[{i}].fc_0_weights not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_1_biases().as_ptr() as *const u8),
            "stack[{i}].fc_1_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_1_weights().as_ptr() as *const u8),
            "stack[{i}].fc_1_weights not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_2_biases().as_ptr() as *const u8),
            "stack[{i}].fc_2_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_2_weights().as_ptr() as *const u8),
            "stack[{i}].fc_2_weights not 64-byte aligned"
        );
    }
}
