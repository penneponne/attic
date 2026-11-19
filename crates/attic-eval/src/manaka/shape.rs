//! The dimensions of the shape this build was compiled for.
//!
//! Nothing else in `manaka` names a width: everything is derived from the
//! constants here, so that a shape of other dimensions is a row in [`DIMS`] and
//! no edit anywhere else.

/// What the shapes differ in. Everything else this module sizes follows from
/// these numbers and the feature space.
struct Dims {
    /// The width of one perspective's embedding.
    dim: usize,
    /// Highest power the accumulator carries.
    order: usize,
    /// The first hidden layer's width.
    hidden: usize,
    /// The second hidden layer's width. NNUE's back half has two layers, and so
    /// does every shape here.
    hidden2: usize,
    /// Channels of the pair table, or 0 for a shape without one. It is a table
    /// of one vector per unordered pair of tokens, so it exists only in the
    /// token space, whose 2,344 rows make 2,745,996 pairs.
    pair: usize,
    /// Channels of the second moment's embedding, or 0 for a shape without one.
    /// The block it feeds is the upper triangle of `Σ v vᵀ`, `m(m+1)/2` long.
    moment: usize,
    /// Channels per square of the square field, or 0 for a shape without one.
    /// The block it feeds is 81 squares of that many channels.
    field: usize,
    /// Whether each hidden layer is read twice, as `clamp(z², 0, 1)` beside `clamp(z, 0, 1)`:
    /// Stockfish's `SqrClippedReLU` and `ClippedReLU` side by side. The second layer then
    /// reads twice the first layer's width, and the value row reads both layers' activations.
    dual: bool,
    /// The dual head's first ablation: the value row reads the second layer's half alone, not
    /// both layers' (`Skip` in the architecture string is gone, and the row is `2·HIDDEN2` wide).
    noskip: bool,
    /// The second ablation: each layer is `[clamp(z, 0, 1) ‖ clamp(z, 0, 1)]`, with no square. The
    /// widths are the dual head's; only the activation is another (`TwinClippedReLU`).
    nosqr: bool,
    /// How far apart two board tokens may be for their pair to have a row that is
    /// read: [`MAX_PAIR_RADIUS`] reads every pair. See [`PAIR_RADIUS`].
    radius: usize,
    /// Whether a token in hand keeps no pair with a board piece that is not a king.
    /// See [`HAND_CUT`].
    hand_cut: bool,
}

impl Dims {
    /// A shape with only the power sums.
    const fn new(dim: usize, order: usize, hidden: usize, hidden2: usize) -> Dims {
        Dims {
            dim,
            order,
            hidden,
            hidden2,
            pair: 0,
            moment: 0,
            field: 0,
            radius: MAX_PAIR_RADIUS,
            hand_cut: false,
            dual: false,
            noskip: false,
            nosqr: false,
        }
    }

    const fn dual(self) -> Dims {
        Dims { dual: true, ..self }
    }

    const fn noskip(self) -> Dims {
        Dims {
            noskip: true,
            ..self
        }
    }

    const fn nosqr(self) -> Dims {
        Dims {
            nosqr: true,
            ..self
        }
    }

    const fn pair(self, pair: usize) -> Dims {
        Dims { pair, ..self }
    }

    const fn moment(self, moment: usize) -> Dims {
        Dims { moment, ..self }
    }

    const fn field(self, field: usize) -> Dims {
        Dims { field, ..self }
    }

    const fn radius(self, radius: usize) -> Dims {
        Dims { radius, ..self }
    }

    const fn hand_cut(self) -> Dims {
        Dims {
            hand_cut: true,
            ..self
        }
    }
}

/// The largest Chebyshev distance two squares of the board can be apart: nine
/// files by nine ranks, so each axis runs 0 to 8. A pair table cut at this radius
/// keeps every pair, which is the table of every shape that is not cut.
pub const MAX_PAIR_RADIUS: usize = 8;

/// One row per shape, or per set of dimensions that the KP, HalfKP and plain
/// token shapes share. The feature space is not here: it is `sym_kp`,
/// `sym_halfkp` or `sym_token` and decides the token count, not these numbers.
/// The shapes with a pair table, a second moment or a square field have no KP or
/// HalfKP counterpart, so each has a row of its own, in the order Manaka's shape
/// code lists the knobs.
///
/// A chain of `cfg!` rather than a `#[cfg]` per row, so that selecting two
/// shapes at once is not a pile of duplicate-definition errors but the one
/// message `types.rs` asserts. The last arm is for `sym` enabled without a
/// shape, which that assertion reports as well.
const DIMS: Dims = if cfg!(any(
    feature = "synn_kp_128x2_32_32",
    feature = "synn_halfkp_128x2_32_32",
    feature = "synn_token_128x2_32_32"
)) {
    Dims::new(128, 3, 32, 32)
} else if cfg!(any(
    feature = "synn_kp_256x2_32_32",
    feature = "synn_token_256x2_32_32"
)) {
    Dims::new(256, 3, 32, 32)
} else if cfg!(feature = "synn_token_128x2_32_32_pair16") {
    Dims::new(128, 3, 32, 32).pair(16)
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_o1_dual_noskip") {
    // The dual head with the value row reading the second layer's half alone.
    Dims::new(128, 1, 32, 32).pair(16).dual().noskip()
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_o1_dual_nosqr") {
    // The dual head with the plain clip in the square's place.
    Dims::new(128, 1, 32, 32).pair(16).dual().nosqr()
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_o1_dual") {
    // The same shape with both hidden layers doubled into a square clip and a plain clip.
    Dims::new(128, 1, 32, 32).pair(16).dual()
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_o1_hc") {
    Dims::new(128, 1, 32, 32).pair(16).hand_cut()
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_o1_r4hc") {
    Dims::new(128, 1, 32, 32).pair(16).radius(4).hand_cut()
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_o1_r4") {
    Dims::new(128, 1, 32, 32).pair(16).radius(4)
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_o1_r2") {
    Dims::new(128, 1, 32, 32).pair(16).radius(2)
} else if cfg!(feature = "synn_token_32x2_32_32_pair32_o1_dual") {
    // The dual head with the embedding narrowed to 32 and the pair table widened to 32 channels.
    Dims::new(32, 1, 32, 32).pair(32).dual()
} else if cfg!(feature = "synn_token_64x2_32_32_pair16_o1_dual") {
    // The dual head with the embedding halved to 64, the pair table as it was.
    Dims::new(64, 1, 32, 32).pair(16).dual()
} else if cfg!(feature = "synn_token_32x2_32_32_pair24_o1_dual") {
    // The dual head with the embedding narrowed to 32 and the pair table widened to 24 channels.
    Dims::new(32, 1, 32, 32).pair(24).dual()
} else if cfg!(any(
    feature = "synn_token_128x2_32_32_pair16_o1",
    feature = "synn_token_128x2_32_32_pair16_o1_hm"
)) {
    // The pair table with the power sums at order one, which is the sum of the
    // embeddings alone: the pair table carries the interactions the higher orders
    // would.
    Dims::new(128, 1, 32, 32).pair(16)
} else if cfg!(feature = "synn_token_128x2_32_32_pair8_o1") {
    // The pair table at half the width: a derive reads half the bytes a pair, the one cost of a node
    // that grows with the pieces on the board.
    Dims::new(128, 1, 32, 32).pair(8)
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_m8_f2") {
    Dims::new(128, 3, 32, 32).pair(16).moment(8).field(2)
} else {
    panic!("a Manaka shape has to be selected: see the features of attic-eval")
};

/// Whether each frame is read in its horizontal-mirror canonical form (`hm`, as HalfKAv2_hm does it): when
/// the frame's own king stands on the sixth to ninth file, every board token of that frame is reflected left
/// to right before its row is read, and the hands are left as they are.
///
/// A property of how the file is read and not of its dimensions, so it is a feature of its own (`sym_hm`) and
/// not a row of [`DIMS`]. It is a constant: a build without it compiles none of the folding, and its code
/// and speed are what they were.
pub const HM: bool = cfg!(feature = "sym_hm");

/// The width of one perspective's embedding.
pub const DIM: usize = DIMS.dim;

/// Highest power the accumulator carries.
pub const ORDER: usize = DIMS.order;

/// The first hidden layer's width.
pub const HIDDEN: usize = DIMS.hidden;

/// The second hidden layer's width.
pub const HIDDEN2: usize = DIMS.hidden2;

/// Whether the hidden layers are doubled: see [`Dims::dual`].
pub const DUAL: bool = DIMS.dual;

/// What the second hidden layer reads: the first layer's activations, which are twice as
/// many when each unit is read twice.
pub const SECOND_INPUT: usize = if DUAL { 2 * HIDDEN } else { HIDDEN };

/// Whether the value row leaves the first layer's activations out: see [`Dims::noskip`].
pub const NOSKIP: bool = DIMS.noskip;

/// Whether the square is taken out of the doubled activations: see [`Dims::nosqr`].
pub const NOSQR: bool = DIMS.nosqr;

/// A dual head's two layers' activations side by side, `[a1 ‖ a2]`, whether or not the value row
/// reads all of it.
pub const DOUBLED: usize = 2 * (HIDDEN + HIDDEN2);

/// Where in `[a1 ‖ a2]` the value row starts reading: after the first layer's half when `noskip`
/// cuts it off.
pub const HEAD_AT: usize = if DUAL && NOSKIP { 2 * HIDDEN } else { 0 };

/// What the value row reads: the last hidden layer, which is the second; or, doubled, both
/// layers' activations side by side, `[a1 ‖ a2]` (the second layer's half alone, `noskip`).
pub const HEAD: usize = if DUAL { DOUBLED - HEAD_AT } else { HIDDEN2 };

const _: () = assert!(
    DUAL || !(NOSKIP || NOSQR),
    "the ablations are of the dual head"
);

/// Rows of the embedding table.
///
/// KP is `fe_end + 2 · 81`. HalfKP is `81 · fe_end + 1` — the `+ 1` is the
/// padding row both kings point at, which YaneuraOu has no equivalent of;
/// see [`crate::features::king_feature`]. Manaka's own space is
/// `14 · 2 · 81 + 2 · 38`: see `tokens.rs`.
pub const TOKENS: usize = crate::types::NUM_FEATURES;

/// Channels of the pair table, or 0 when the shape has none.
pub const PAIR: usize = DIMS.pair;

/// How far apart two board tokens may be, in squares of the larger of the file and
/// rank differences, for their pair's row to be read. A pair that touches a hand
/// token or a king is always read. [`MAX_PAIR_RADIUS`], which is every shape's but
/// the `_r` ones, reads every pair.
///
/// A pair the radius cuts is not in the network: the trainer's
/// `--pair-radius` does not read it, does not train it and writes zeros for it,
/// so a net trained at a radius has no information in the rows this skips, and a
/// build at a radius other than its net's would read rows that were never trained
/// or skip ones that were. Nothing in the file says which radius it was trained
/// at; the shape does.
pub const PAIR_RADIUS: usize = DIMS.radius;

/// Whether a token in hand keeps no pair with a board piece that is not a king: the
/// trainer's `--pair-hand-cut`. The held piece is still a token with a row of the
/// embedding; what is gone is its pairs with the board. Its pairs with a king and
/// with another held piece are kept. Like the radius, the net does not say.
pub const HAND_CUT: bool = DIMS.hand_cut;

/// Whether the shape cuts its pair table: a constant, so that a shape that does
/// not has none of the filter in its build.
pub const PAIR_CUT: bool = PAIR_RADIUS < MAX_PAIR_RADIUS || HAND_CUT;

/// Channels of the second moment's embedding, or 0 when the shape has none.
pub const MOMENT: usize = DIMS.moment;

/// Channels per square of the square field, or 0 when the shape has none.
pub const FIELD: usize = DIMS.field;

/// Whether the shape has a pair table.
///
/// A constant, not a field of the network, so that a shape without one tests a
/// number the compiler knows: the pair sum, its table and its block vanish from
/// that build. Spelled `!= 0` and not `> 0` because clippy rejects the latter on
/// the shapes where it is always false, which is the point of it being a
/// constant.
pub const HAS_PAIR: bool = PAIR != 0;

/// Rows of the pair table: one per unordered pair of distinct tokens, which is
/// `C(TOKENS, 2)`. Zero without a pair table, since in the KP and HalfKP spaces
/// the same formula would name 7.9e9 rows that nothing reads.
pub const PAIR_ROWS: usize = if HAS_PAIR {
    TOKENS * (TOKENS - 1) / 2
} else {
    0
};

/// Whether the shape has a second moment. Spelled `!= 0` for the reason
/// [`HAS_PAIR`] is.
pub const HAS_MOMENT: bool = MOMENT != 0;

/// Whether the shape has a square field.
pub const HAS_FIELD: bool = FIELD != 0;

/// The squares a field has a slice for: the board's. The reader never uses what
/// a slice means, only that there are 81 of them.
pub(super) const SQUARES: usize = attic_state::Square::COUNT;

/// Entries of the second moment: the upper triangle of an `MOMENT × MOMENT`
/// matrix, diagonal included, which is what the head reads. The rest is the same
/// numbers again, since `Σ v vᵀ` is symmetric.
pub const MOMENT_LEN: usize = MOMENT * (MOMENT + 1) / 2;

/// Entries of the square field: a slice of `FIELD` channels for each square.
pub const FIELD_LEN: usize = SQUARES * FIELD;

/// The power sums one perspective carries: every order.
pub const POWER_LEN: usize = ORDER * DIM;

/// One perspective's share of what the first hidden layer reads, laid out as
/// the power blocks, then the pair channels, the second moment and the square
/// field. A block that a shape lacks is zero long, so its offset is where the
/// next one starts.
pub const BLOCK_LEN: usize = POWER_LEN + PAIR + MOMENT_LEN + FIELD_LEN;

/// Where the pair channels start within a block.
pub const PAIR_AT: usize = POWER_LEN;

/// Where the second moment starts within a block.
pub const MOMENT_AT: usize = PAIR_AT + PAIR;

/// Where the square field starts within a block.
pub const FIELD_AT: usize = MOMENT_AT + MOMENT_LEN;

/// What the first hidden layer reads: both perspectives' blocks.
pub const FEATURE_LEN: usize = 2 * BLOCK_LEN;

// Exactly one shape, counted. `DIMS` above picks its row by testing the shapes in
// turn, so two at once would silently take the first. `sym` on its own, which no
// shape is, counts zero and fails here too, which is why the guard in `types.rs`
// can stand for every one of them with `sym`.
#[cfg(feature = "sym")]
const _: () = {
    macro_rules! selected {
        ($($shape:literal),+ $(,)?) => { 0 $(+ cfg!(feature = $shape) as usize)+ };
    }
    let selected = selected!(
        "synn_kp_128x2_32_32",
        "synn_kp_256x2_32_32",
        "synn_halfkp_128x2_32_32",
        "synn_token_128x2_32_32",
        "synn_token_256x2_32_32",
        "synn_token_128x2_32_32_pair16",
        "synn_token_128x2_32_32_pair16_o1",
        "synn_token_128x2_32_32_pair16_o1_dual",
        "synn_token_128x2_32_32_pair16_o1_dual_noskip",
        "synn_token_128x2_32_32_pair16_o1_dual_nosqr",
        "synn_token_128x2_32_32_pair16_o1_hm",
        "synn_token_128x2_32_32_pair8_o1",
        "synn_token_128x2_32_32_pair16_o1_r4",
        "synn_token_128x2_32_32_pair16_o1_hc",
        "synn_token_128x2_32_32_pair16_o1_r4hc",
        "synn_token_128x2_32_32_pair16_o1_r2",
        "synn_token_32x2_32_32_pair32_o1_dual",
        "synn_token_64x2_32_32_pair16_o1_dual",
        "synn_token_32x2_32_32_pair24_o1_dual",
        "synn_token_128x2_32_32_pair16_m8_f2",
    );
    assert!(
        selected == 1,
        "select exactly one Manaka shape: they differ in their dimensions"
    );
};

const _: () = {
    // Only the token space has an own king whose file can be asked, and a table row that a fold can reflect.
    assert!(
        !HM || cfg!(feature = "sym_token"),
        "hm folds the board squares of the token space only"
    );
};

const _: () = {
    // NNUE's back half has two dense layers, and the file layout, the architecture
    // string and the forward pass all assume both.
    assert!(
        HIDDEN2 != 0,
        "a Manaka shape has two hidden layers, as NNUE's back half does"
    );
    // Only the token space is small enough for a row per pair, and the loader
    // and the accumulator index the table by a token pair directly.
    assert!(
        !HAS_PAIR || cfg!(feature = "sym_token"),
        "a pair table exists only in the token space"
    );
    // A radius cuts the pairs of a pair table, so there has to be one, and it is
    // a distance between two squares of the board.
    assert!(
        PAIR_RADIUS <= MAX_PAIR_RADIUS && (!PAIR_CUT || HAS_PAIR),
        "a pair cut needs a pair table, and a radius is at most the board's width less one"
    );
    // The moment and the field want a row per token, as the embedding does, and
    // the other spaces have up to 125,389 of them: a field of eight channels per
    // square would be 81 million numbers in the HalfKP space. Manaka's shape
    // grammar refuses both outside the token space for that reason, and so does
    // this.
    assert!(
        !(HAS_MOMENT || HAS_FIELD) || cfg!(feature = "sym_token"),
        "a second moment or a square field exists only in the token space"
    );
};
