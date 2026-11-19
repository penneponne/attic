//! What Manaka's exporter wrote for each shape's architecture string.
//!
//! Kept apart from the fixtures because it grows by a constant for every shape,
//! and because it is a table of facts and not code: each is copied from a real
//! export, never produced by this crate.

/// What Manaka's exporter (`sym_arch_string`) wrote for this shape: the string
/// after its leading `Features=`, and the file hash it stored. Copied from the
/// header of the file `manaka-train export-nnue` wrote for a net of each shape (at
/// tideborn's commit 91421804): asserting a function against its own output passes
/// for any string, and these are what the exporter said.
///
/// A new shape is a new row, from a new export.
pub(super) struct Exported {
    /// Everything after `Features=`.
    pub(super) rest: &'static str,
    pub(super) hash: u32,
}

#[cfg(feature = "synn_token_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    hash: 0x263525dd,
};

#[cfg(feature = "synn_kp_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    hash: 0xe7c9e6dd,
};

/// From the file `export-nnue` wrote at tideborn's develop e6141b7d for a net
/// trained at this shape (2026-11-11), not at 91421804 like the rest: the shape
/// is newer than that commit. The SYNN string is assembled from the shape there,
/// so the exporter needed no change to write it.
#[cfg(feature = "synn_kp_256x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->256x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-1536](ClipScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))))",
    hash: 0x7b28b1e1,
};

/// From the file `export-nnue` wrote at tideborn's develop 91421804 for this
/// shape's seed-1 initial net (2026-11-11); the string depends on the shape alone.
#[cfg(feature = "synn_token_256x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[256x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-1536](ClipScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))))",
    hash: 0xbdfcdee1,
};

#[cfg(feature = "synn_halfkp_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    hash: 0xe1e6babf,
};

#[cfg(feature = "synn_token_128x2_32_32_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-800](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),ClipScaled[16](PairSum[16](InputSlice[16(384:400)]))))))",
    hash: 0x25faf650,
};

/// From the file `export-nnue` wrote at tideborn's develop 91421804 for a net
/// trained at this shape (QS 1000M, 2026-11-11); the string depends on the shape
/// alone. At order one `Newton[1]` is the identity, and the exporter writes it all
/// the same.
#[cfg(feature = "synn_token_128x2_32_32_pair16_o1")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-288](ClipScaled[128](Newton[1](PowerSum[1](InputSlice[128(0:128)]))),ClipScaled[16](PairSum[16](InputSlice[16(128:144)]))))))",
    hash: 0x4fecfd41,
};

#[cfg(feature = "synn_token_128x2_32_32_pair16_m8_f2")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-1196](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),ClipScaled[16](PairSum[16](InputSlice[16(384:400)])),ClipScaled[36](SecondMoment[8](InputSlice[36(400:436)])),ClipScaled[162](SquareField[2](InputSlice[162(436:598)]))))))",
    hash: 0x00191180,
};

/// From the file `manaka-train export-nnue` wrote for this shape's seed-1 initial net
/// (`pair16_d128_o1_h32_h232_dual`, tideborn branch `feat/synn-dual-act`, 2026-11-12); the
/// string depends on the shape alone. The two hidden layers are `DualClippedReLU` — a square
/// clip beside a plain one — the second layer reads `2 · 32`, and `Skip` joins the activations
/// of both layers into the value row, which is `2 · (32 + 32)` wide.
#[cfg(feature = "synn_token_128x2_32_32_pair16_o1_dual")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-128](Skip[32,32](DualClippedReLU[32](Dense[32<-64](DualClippedReLU[32](Dense[32<-288](ClipScaled[128](Newton[1](PowerSum[1](InputSlice[128(0:128)]))),ClipScaled[16](PairSum[16](InputSlice[16(128:144)])))))))",
    hash: 0xc450c6de,
};

/// The first ablation of the dual head, from the file `export-nnue` wrote for its seed-1 initial net
/// (`pair16_d128_o1_h32_h232_dual_noskip`, tideborn `0694e7b5`, 2026-11-12). There is no `Skip`: the value
/// row reads the second layer's `[sqr || clip]` alone, 64 wide.
#[cfg(feature = "synn_token_128x2_32_32_pair16_o1_dual_noskip")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-64](DualClippedReLU[32](Dense[32<-64](DualClippedReLU[32](Dense[32<-288](ClipScaled[128](Newton[1](PowerSum[1](InputSlice[128(0:128)]))),ClipScaled[16](PairSum[16](InputSlice[16(128:144)]))))))",
    hash: 0xf4d918d5,
};

/// The second ablation, from the same export of `pair16_d128_o1_h32_h232_dual_nosqr`: the dual head's
/// widths and `Skip`, with `TwinClippedReLU` (a plain clip twice) where the dual head has the square.
#[cfg(feature = "synn_token_128x2_32_32_pair16_o1_dual_nosqr")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-128](Skip[32,32](TwinClippedReLU[32](Dense[32<-64](TwinClippedReLU[32](Dense[32<-288](ClipScaled[128](Newton[1](PowerSum[1](InputSlice[128(0:128)]))),ClipScaled[16](PairSum[16](InputSlice[16(128:144)])))))))",
    hash: 0xad8bcc76,
};

/// From the file `export-nnue` wrote for a net trained at this shape with `hm` (QS 1000M, 2026-11-12,
/// tideborn `feat/synn-dual-act` ee7bf590): the string is the plain pair16 o1's with `FoldedTokens` where it
/// says `Tokens`, and nothing else. A build without hm refuses this string and this build refuses the plain one.
#[cfg(feature = "synn_token_128x2_32_32_pair16_o1_hm")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "FoldedTokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-288](ClipScaled[128](Newton[1](PowerSum[1](InputSlice[128(0:128)]))),ClipScaled[16](PairSum[16](InputSlice[16(128:144)]))))))",
    hash: 0x0af15d6f,
};

/// From the file `export-nnue` wrote for a net at this shape (tideborn `feat/hm-pair-plain` d7bf3a4d,
/// 2026-11-18): the pair16 o1 string with the pair table eight wide.
#[cfg(feature = "synn_token_128x2_32_32_pair8_o1")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-272](ClipScaled[128](Newton[1](PowerSum[1](InputSlice[128(0:128)]))),ClipScaled[8](PairSum[8](InputSlice[8(128:136)]))))))",
    hash: 0x185437be,
};
