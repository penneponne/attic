//! The accumulator the search threads through the tree, and what a derive needs.

use attic_state::{Color, Position};

#[cfg(not(feature = "sym_token"))]
use crate::features::{FeatureIndex, MoveDelta, active_features, active_features_into};
#[cfg(feature = "sym_token")]
use crate::tokens::{
    FeatureIndex, MAX_ACTIVE_FEATURES, MoveDelta, PerspectiveDelta, active_features,
};

use super::cache::FinnyCache;
#[cfg(feature = "sym_token")]
use super::cache::{FoldEntry, TokenList};
#[cfg(feature = "sym_token")]
use super::field::touch_field;
#[cfg(feature = "sym_token")]
use super::moment::touch_moment;
#[cfg(feature = "sym_token")]
use super::pair::{touch_pairs, touch_pairs_hm};
#[cfg(feature = "sym_token")]
use super::shape::{HAS_FIELD, HAS_MOMENT, HAS_PAIR};
#[cfg(feature = "sym_token")]
use super::shape::{HM, HM_NOPAIRFOLD};
use super::{DIM, FIELD_LEN, MOMENT_LEN, ORDER, PAIR, POWER_LEN, SymNetwork};

/// What the pair sums are accumulated in. `f64`, which is not what the table is
/// stored in (halves) or the forward pass computes in (`f32`): see the note on
/// [`Accumulator`].
pub(super) type PairSum = f64;

/// What the second moment's sums are accumulated in.
pub(super) type MomentSum = f64;

/// What the square field's sums are accumulated in.
pub(super) type FieldSum = f64;

/// The accumulator the search threads through the tree.
///
/// Holds each perspective's power sums, and in the token shapes the tokens they
/// were built from.
///
/// ## Why the sums update incrementally and the polynomials do not
///
/// `p_k[c] = Σ_i v_i[c]^k` is a sum over tokens, so a token leaving subtracts
/// its own `k`-th powers and a token arriving adds them — the other
/// thirty-eight are untouched. The elementary symmetric polynomials that
/// [`elementary_from_power`] derives are *not* sums over tokens and cannot be
/// updated this way, which is exactly why the accumulator stores `p` and the
/// evaluation converts. That conversion is `ORDER · DIM` multiply-adds once
/// per evaluation rather than per token, and it is the whole reason Newton's
/// identities are in the forward at all.
///
/// ## How a derive gets from one position to the next
///
/// The search hands every derive the accumulator of the position's parent and
/// the `MoveDelta` of the move between them, and that names the (at most) two
/// features that leave and the two that arrive without looking at the board.
/// Every shape applies it to the parent's sums. The KP and HalfKP shapes used to
/// scan the new position and diff its list against one they held instead, which
/// found the same four or so features and cost a scan and two sorts per node,
/// more than the rest of the derive put together.
///
/// The one move a delta cannot describe is HalfKP's own-king move: every
/// feature of that half is crossed with the king's square, so all forty change.
/// That half is rebuilt from the position after the move, which is forty rows
/// where a diff of the two lists would have been seventy-six.
///
/// ## The token shapes' set of tokens
///
/// In the token shapes `active` is kept up to date by the same delta, so that
/// anything that needs the other tokens present — a pair table pairs each
/// changed token with them — has the set it needs at the point it needs it: a
/// derive takes each removed token out of `active` *before* it subtracts the
/// token's row, and puts each added one in *after* it adds the row.
///
/// ## The pair table
///
/// A pair table is the one sum here that is over *pairs* of tokens, so a token
/// that arrives adds one row for every token already present and one that
/// leaves subtracts the same rows. That is why `active` is kept, and why the
/// order above matters: a capture takes the victim out, then the lifted piece,
/// and the lifted piece's rows against the victim are gone with the victim's.
///
/// Both frames' pair sums are moved by one call, from the black frame's tokens:
/// a row of the table holds both frames' channels for a pair (`pair_table.rs`).
///
/// Each of those is a floating-point sum taken out of a larger one, which is
/// where `f32` drifts, so the pair sum is accumulated in `f64`. A token's rows,
/// at most thirty-nine, are added up in `f32` first and go into it as one total
/// (`touch_pairs`), so the `f64` sum takes one rounding per token touched. The
/// search seeds the root with [`Accumulator::refresh`] every iteration, which is
/// what keeps the drift to a few plies' worth; see `tests/sym_network.rs` for how
/// much that is on a trained net.
#[derive(Debug, Clone)]
pub struct Accumulator {
    /// `[perspective][POWER_LEN]`, indexed by [`Color::index`]; within one
    /// perspective `[order][channel]`.
    sums: [Vec<f32>; Color::COUNT],
    /// `[perspective][PAIR]`: the sum over the pair table of every pair of
    /// tokens present. Empty for a shape without one.
    pair: [Vec<PairSum>; Color::COUNT],
    /// `[perspective][MOMENT_LEN]`: the upper triangle of `Σ v vᵀ` over the
    /// tokens present. Empty for a shape without a second moment.
    moment: [Vec<MomentSum>; Color::COUNT],
    /// `[perspective][FIELD_LEN]`: the sum of the present tokens' rows of the
    /// field, `[square][channel]`. Empty for a shape without a field.
    field: [Vec<FieldSum>; Color::COUNT],
    /// The tokens present in each half, which a table over pairs of tokens is
    /// read against. The KP and HalfKP shapes have no such table and keep no
    /// list: the delta names what changes, and a half that is rebuilt is
    /// rebuilt from the position.
    #[cfg(feature = "sym_token")]
    active: [Vec<FeatureIndex>; Color::COUNT],
    /// Whether each frame is read mirrored (`hm`): its own king is on the sixth to ninth file. The sums above
    /// are those of the tokens `active` holds *after* this reflection, and `active` itself is kept as the
    /// frame numbers its tokens, unreflected, so that a frame that has to be built again (its king crossed the
    /// fifth file) is built from what is on the board. Always `false` in a build without `hm`.
    #[cfg(feature = "sym_token")]
    folded: [bool; Color::COUNT],
    /// How many tokens each frame's pair sum has been moved by since it was last summed from nothing, which
    /// is how far it can have drifted from that sum: each touch adds a total rounded to `f32`. A frame
    /// rebuilt from the state it last had in a fold ([`FinnyCache`]) carries that state's count on, so that
    /// a sum kept across many crossings is summed from nothing again once it passes [`LINEAGE_LIMIT`].
    #[cfg(feature = "sym_token")]
    lineage: [u32; Color::COUNT],
}

/// The most touches a pair sum taken from [`FinnyCache`] may carry: past it, the frame is summed from nothing.
/// A search's root is refreshed every iteration, and a line from it moves a sum by a few tokens a ply, so this
/// keeps a sum that crossed the line to about what one deep line already has.
#[cfg(feature = "sym_token")]
const LINEAGE_LIMIT: u32 = 256;

/// The folds the pair table is read in: each frame's own under `hm`, and neither under [`HM_NOPAIRFOLD`],
/// whose pairs are read at the tokens as they stand whatever the frames' folds.
#[cfg(feature = "sym_token")]
#[inline]
fn pair_folds(folded: [bool; Color::COUNT]) -> [bool; Color::COUNT] {
    if HM_NOPAIRFOLD {
        [false; Color::COUNT]
    } else {
        folded
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            sums: [vec![0.0; POWER_LEN], vec![0.0; POWER_LEN]],
            pair: [vec![0.0; PAIR], vec![0.0; PAIR]],
            moment: [vec![0.0; MOMENT_LEN], vec![0.0; MOMENT_LEN]],
            field: [vec![0.0; FIELD_LEN], vec![0.0; FIELD_LEN]],
            #[cfg(feature = "sym_token")]
            active: [
                Vec::with_capacity(MAX_ACTIVE_FEATURES),
                Vec::with_capacity(MAX_ACTIVE_FEATURES),
            ],
            #[cfg(feature = "sym_token")]
            folded: [false; Color::COUNT],
            #[cfg(feature = "sym_token")]
            lineage: [0; Color::COUNT],
        }
    }

    /// Rebuild both perspectives from scratch.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king.
    pub fn refresh(&mut self, net: &SymNetwork, pos: &Position) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            let features = active_features(pos, color);
            #[cfg(feature = "sym_token")]
            {
                self.folded[i] = HM && crate::tokens::folds(&features);
                self.lineage[i] = 0;
                self.pair[i].fill(0.0);
                self.moment[i].fill(0.0);
                self.field[i].fill(0.0);
                self.sum_frame(net, i, &features);
                self.active[i] = features;
            }
            #[cfg(not(feature = "sym_token"))]
            {
                accumulate(&ZERO_SUMS, &mut self.sums[i], &net.embed, &[], &features);
                self.pair[i].fill(0.0);
                self.moment[i].fill(0.0);
                self.field[i].fill(0.0);
            }
        }
        // Every pair once, each token against the ones before it, both frames at
        // once from the black frame's list.
        #[cfg(feature = "sym_token")]
        {
            let black = std::mem::take(&mut self.active[Color::Black.index()]);
            let folded = pair_folds(self.folded);
            for (at, &token) in black.iter().enumerate() {
                let others = &black[..at];
                if HM {
                    touch_pairs_hm(
                        &mut self.pair,
                        net,
                        token,
                        others,
                        1.0,
                        folded,
                        [true, true],
                    );
                } else {
                    touch_pairs(&mut self.pair, net, token, others, 1.0);
                }
            }
            self.active[Color::Black.index()] = black;
        }
    }

    /// The power sums, the second moment and the square field of frame `i` built from `tokens`, as that
    /// frame numbers them and unreflected: the frame's fold (`self.folded[i]`) is applied here, so that every
    /// caller states what is on the board and this decides which rows that is. The pair sum is not touched;
    /// it is over pairs and is the callers' to build, in the order they need.
    #[cfg(feature = "sym_token")]
    fn sum_frame(&mut self, net: &SymNetwork, i: usize, tokens: &[FeatureIndex]) {
        let mut buffer = [0 as FeatureIndex; MAX_ACTIVE_FEATURES];
        let rows: &[FeatureIndex] = if self.folded[i] {
            for (slot, &token) in buffer.iter_mut().zip(tokens) {
                *slot = crate::tokens::mirror_row(token);
            }
            &buffer[..tokens.len()]
        } else {
            tokens
        };
        accumulate(&ZERO_SUMS, &mut self.sums[i], &net.embed, &[], rows);
        for &row in rows {
            touch_moment(&mut self.moment[i], net, row, 1.0);
            touch_field(&mut self.field[i], net, row, 1.0);
        }
    }

    /// The power sums for one perspective, `[order][channel]`.
    pub fn perspective(&self, color: Color) -> &[f32] {
        &self.sums[color.index()]
    }

    /// The pair table's sum for one perspective, `[channel]`; empty without one.
    ///
    /// Accumulated in `f64` and handed over rounded to `f32`, which is what the
    /// forward pass computes in; see [`Accumulator`].
    pub fn pair(&self, color: Color) -> [f32; PAIR] {
        std::array::from_fn(|channel| self.pair[color.index()][channel] as f32)
    }

    /// The second moment's upper triangle for one perspective, `[entry]`; empty
    /// without one. Rounded to `f32` from the `f64` it is accumulated in.
    pub fn moment(&self, color: Color) -> [f32; MOMENT_LEN] {
        std::array::from_fn(|entry| self.moment[color.index()][entry] as f32)
    }

    /// The square field's sum for one perspective, `[square][channel]` flattened;
    /// empty without one. Rounded to `f32` from the `f64` it is accumulated in.
    pub fn field(&self, color: Color) -> [f32; FIELD_LEN] {
        std::array::from_fn(|at| self.field[color.index()][at] as f32)
    }

    /// Which frames are read mirrored; the tests state what the fold did from this.
    #[cfg(all(test, feature = "sym_hm"))]
    pub(super) fn folded_for_test(&self) -> [bool; Color::COUNT] {
        self.folded
    }

    /// The tokens present in one perspective, in no particular order.
    #[cfg(all(test, feature = "sym_token"))]
    pub(super) fn tokens(&self, color: Color) -> &[FeatureIndex] {
        &self.active[color.index()]
    }
}

/// KP and HalfKP: by the move's delta, from the parent's accumulator, with a
/// half rebuilt where the delta has none.
#[cfg(not(feature = "sym_token"))]
impl Accumulator {
    /// Derive the accumulator for the position after `mv` from `self`, the one
    /// before it.
    ///
    /// `pos` is the pre-move position and is left byte-for-byte unchanged: the
    /// do/undo happens inside, because a half that is rebuilt is rebuilt from
    /// the position after the move. Mirrors the NNUE accumulator's contract so
    /// that the search treats the two the same way.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king, before or after `mv`.
    pub fn update_after_move(
        &self,
        net: &SymNetwork,
        pos: &mut Position,
        mv: attic_state::Move,
    ) -> Accumulator {
        let delta = MoveDelta::from_move(pos, mv);
        let undo = pos.do_move(mv);
        let mut next = Accumulator::new();
        // The one call site that has no cache to borrow from — the root, once
        // per iteration. The tree goes through `derive_into_cached`.
        let mut cache = FinnyCache::default();
        Self::derive_into_cached(self, &mut next, net, pos, &delta, &mut cache);
        pos.undo_move(mv, undo);
        next
    }

    /// Write into `dst` the accumulator of the position after the move `delta`
    /// describes, from `src`, which has to be the accumulator of the position the
    /// move was made from.
    ///
    /// Each half of `dst` is written from `src` in one pass, the parent's sums
    /// read, the move's rows taken out and put in, and the result stored: there is
    /// no copy of the parent first. Whatever `dst` held is overwritten.
    ///
    /// The delta is NNUE's, and the features it names are the ones the scan of a
    /// position would list: both read `features.rs`, which numbers K+P and HalfKP
    /// for every shape that uses them (its tests hold `MoveDelta::from_move` to
    /// the scan on every legal move of a spread of positions). A KP half always
    /// has one, two removed and two added at most. A HalfKP half has none when
    /// its own king moved, because every feature in it is crossed with that
    /// king's square: that half is summed again from the forty features of
    /// `post_pos`, through `cache`'s list so that the rebuild does not allocate.
    /// That is the only time `post_pos` and `cache` are read.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        post_pos: &Position,
        delta: &MoveDelta,
        cache: &mut FinnyCache,
    ) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            match delta.half(color) {
                Some(half) => accumulate(
                    &src.sums[i],
                    &mut dst.sums[i],
                    &net.embed,
                    half.removed(),
                    half.added(),
                ),
                None => {
                    active_features_into(post_pos, color, &mut cache.features);
                    accumulate(
                        &ZERO_SUMS,
                        &mut dst.sums[i],
                        &net.embed,
                        &[],
                        &cache.features,
                    );
                }
            }
        }
    }
}

// The KP and HalfKP spaces have no pair table, second moment or square field
// (Manaka's shape grammar refuses them there), so the power sums are all a
// derive writes: the other three are empty in both accumulators.
#[cfg(not(feature = "sym_token"))]
const _: () = assert!(
    PAIR == 0 && MOMENT_LEN == 0 && FIELD_LEN == 0,
    "a KP or HalfKP shape carries nothing but its power sums"
);

/// The token shapes: by the move's delta, from the parent's accumulator.
#[cfg(feature = "sym_token")]
impl Accumulator {
    /// Derive the accumulator for the position after `mv` from `self`, the one
    /// before it.
    ///
    /// `pos` is the pre-move position and is not changed; it is `&mut` for the
    /// signature the other shapes have, where the move is done and undone
    /// inside.
    pub fn update_after_move(
        &self,
        net: &SymNetwork,
        pos: &mut Position,
        mv: attic_state::Move,
    ) -> Accumulator {
        let delta = MoveDelta::from_move(pos, mv);
        let mut next = Accumulator::new();
        Self::derive_into_cached(
            self,
            &mut next,
            net,
            pos,
            &delta,
            &mut FinnyCache::default(),
        );
        next
    }

    /// Write into `dst` the accumulator of the position after the move `delta`
    /// describes, from `src`, which has to be the accumulator of the position the
    /// move was made from. Whatever `dst` held is overwritten.
    ///
    /// The position and the cache go unread: a token's row does not depend on
    /// any other piece, so the delta is everything, and there is no bucket whose
    /// rebuild a cache could hold.
    ///
    /// The power sums are written from `src` in one pass, with no copy of the
    /// parent's first. The pair, moment and field sums and the set of tokens are
    /// copied and then changed, token by token: each removal leaves `active`
    /// before its row is subtracted and each addition enters it after its row is
    /// added, so at the moment a token's row is touched `active` is the set of
    /// *other* tokens present — what a table over pairs of tokens must read to
    /// subtract or add the right rows.
    ///
    /// # Panics
    /// Panics if a removed token is not present, which means `src` was not the
    /// accumulator of the position the move was made from.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        _post_pos: &Position,
        delta: &MoveDelta,
        cache: &mut FinnyCache,
    ) {
        let halves = [Color::Black, Color::White]
            .map(|color| delta.half(color).expect("a token delta is never a rebuild"));
        if HM {
            Self::derive_hm(src, dst, net, &halves, cache);
            return;
        }
        for (i, half) in halves.iter().enumerate() {
            // The power sums depend on nothing but the rows being taken out and put
            // in, and in what order, so they go in one pass ahead of the tables that
            // read `active`.
            accumulate(
                &src.sums[i],
                &mut dst.sums[i],
                &net.embed,
                half.removed(),
                half.added(),
            );
            dst.copy_sums_from(src, i);
        }
        let [black, white] = halves;
        dst.touch_tokens(net, black, white);
    }

    /// [`Self::derive_into_cached`] for a net read with `hm`.
    ///
    /// Each frame is read in its own fold, mirrored when its own king is on the sixth to ninth file, and a
    /// move can change that for one frame and not the other. A frame whose fold holds still is derived as
    /// ever, except that the rows it takes out and puts in are the reflected ones if it is folded. A frame
    /// whose own king crossed the line between the fifth file and the sixth is read the other way now, which
    /// changes every board token of it, so it is built again from the tokens on the board after the move:
    /// forty rows of the embedding, the same of the moment and the field, and its pair sum. That is about one
    /// derive in twenty-five in a search, and every other derive is the incremental update it was.
    ///
    /// The pair sum is the costly part, a row for each of the 780 pairs, and it is not built from nothing
    /// unless it has to be. `cache` holds the state each frame was last in under each fold; the frame's pair
    /// sum is that state's, moved by the tokens the boards differ in, which is a dozen or so where a sum from
    /// nothing is forty. The state the frame leaves is kept for its return, and the state it enters is kept
    /// too, so that a sibling move crossing the same way starts from it. A frame is summed from nothing when
    /// there is no state for the fold, when the boards differ in so many tokens that the moves would read more
    /// rows than the sum does, or when the state has been moved by more than [`LINEAGE_LIMIT`] tokens.
    ///
    /// Under [`HM_NOPAIRFOLD`] the pair table is read at the tokens as they stand, so a frame's pair sum does
    /// not depend on its fold: both frames' are moved by the move's tokens as a net without `hm` moves them,
    /// a crossing rebuilds the embedding, the moment and the field and nothing of the table, and `cache` is
    /// not read.
    ///
    /// The set of tokens `active` is the board as each frame numbers it, never reflected; the reflection is
    /// applied where the rows are read.
    fn derive_hm(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        halves: &[&PerspectiveDelta; Color::COUNT],
        cache: &mut FinnyCache,
    ) {
        use crate::tokens::{folds_for_king, is_own_king, mirror_row};

        // The fold of each frame after the move: an own king among the tokens it gains says where it
        // stands now, and a frame that gains none keeps the fold it had.
        let mut rebuild = [false; Color::COUNT];
        for (i, half) in halves.iter().enumerate() {
            dst.folded[i] = src.folded[i];
            if let Some(&king) = half.added().iter().find(|&&token| is_own_king(token)) {
                dst.folded[i] = folds_for_king(king);
            }
            rebuild[i] = dst.folded[i] != src.folded[i];
        }

        // The power sums, the moment and the field of the frames that hold their fold, from the parent in one
        // pass with the rows reflected if the frame is folded. A frame that rebuilds is written below.
        for (i, half) in halves.iter().enumerate() {
            dst.copy_sums_from(src, i);
            if rebuild[i] {
                continue;
            }
            dst.lineage[i] =
                src.lineage[i].saturating_add((half.removed().len() + half.added().len()) as u32);
            if !src.folded[i] {
                accumulate(
                    &src.sums[i],
                    &mut dst.sums[i],
                    &net.embed,
                    half.removed(),
                    half.added(),
                );
                continue;
            }
            let (mut removed, mut added) = ([0 as FeatureIndex; 2], [0 as FeatureIndex; 2]);
            for (slot, &token) in removed.iter_mut().zip(half.removed()) {
                *slot = mirror_row(token);
            }
            for (slot, &token) in added.iter_mut().zip(half.added()) {
                *slot = mirror_row(token);
            }
            accumulate(
                &src.sums[i],
                &mut dst.sums[i],
                &net.embed,
                &removed[..half.removed().len()],
                &added[..half.added().len()],
            );
        }

        // The tokens, the pair sums and the moment and field of the frames that hold their fold, token by
        // token as in the unfolded derive: each removal leaves `active` before its pairs are taken out, and
        // each addition enters it after its pairs are put in. A pair table read unfolded does not care what
        // fold a frame is in, so under `HM_NOPAIRFOLD` both frames' pair sums are moved here, crossed or not.
        let keep = [!rebuild[0], !rebuild[1]];
        let (pair_fold, pair_keep) = match HM_NOPAIRFOLD {
            true => ([false; Color::COUNT], [true; Color::COUNT]),
            false => (src.folded, keep),
        };
        let (black, white) = (halves[0], halves[1]);
        let (b, w) = (Color::Black.index(), Color::White.index());
        for (&token, &other) in black.removed().iter().zip(white.removed()) {
            for (i, token) in [(b, token), (w, other)] {
                let at = position_of(&dst.active[i], token)
                    .expect("a removed token was not in the accumulator it is removed from");
                dst.active[i].swap_remove(at);
            }
            touch_pairs_hm(
                &mut dst.pair,
                net,
                token,
                &dst.active[b],
                -1.0,
                pair_fold,
                pair_keep,
            );
            for (i, token) in [(b, token), (w, other)] {
                if keep[i] {
                    let row = if src.folded[i] {
                        mirror_row(token)
                    } else {
                        token
                    };
                    touch_moment(&mut dst.moment[i], net, row, -1.0);
                    touch_field(&mut dst.field[i], net, row, -1.0);
                }
            }
        }
        for (&token, &other) in black.added().iter().zip(white.added()) {
            touch_pairs_hm(
                &mut dst.pair,
                net,
                token,
                &dst.active[b],
                1.0,
                pair_fold,
                pair_keep,
            );
            for (i, token) in [(b, token), (w, other)] {
                if keep[i] {
                    let row = if src.folded[i] {
                        mirror_row(token)
                    } else {
                        token
                    };
                    touch_moment(&mut dst.moment[i], net, row, 1.0);
                    touch_field(&mut dst.field[i], net, row, 1.0);
                }
                dst.active[i].push(token);
            }
        }

        // A frame whose fold changed is built again from the board it now reads.
        if rebuild.contains(&true) && !HM_NOPAIRFOLD {
            cache.for_net(net);
        }
        for (i, _) in rebuild.iter().enumerate().filter(|(_, crossed)| **crossed) {
            dst.moment[i].fill(0.0);
            dst.field[i].fill(0.0);
            let tokens = std::mem::take(&mut dst.active[i]);
            dst.sum_frame(net, i, &tokens);
            dst.active[i] = tokens;
            // Its pair sum was moved by the move above, as the unfolded table it is read in is.
            if HM_NOPAIRFOLD {
                continue;
            }

            // The state the frame leaves, in the fold it leaves, for when its king comes back.
            cache.folds[i][usize::from(src.folded[i])] = Some(FoldEntry {
                tokens: TokenList::black(&src.active[i], i),
                pair: src.pair[i]
                    .as_slice()
                    .try_into()
                    .expect("a pair sum is PAIR long"),
                lineage: src.lineage[i],
            });
            // The pair sum it enters, read in the black frame's numbering (`touch_pairs_hm` reads the table in
            // it), and the state it now has kept for a sibling that crosses the same way.
            let now = TokenList::black(&dst.active[i], i);
            let entering = cache.folds[i][usize::from(dst.folded[i])];
            let built = Self::pair_from(net, i, dst.folded, entering.as_ref(), &now, cache);
            dst.pair[i].copy_from_slice(&built.pair);
            dst.lineage[i] = built.lineage;
            cache.folds[i][usize::from(dst.folded[i])] = Some(built);
        }
    }

    /// Frame `i`'s pair sum over `now` (the black frame's numbering) in the folds `folded`: from `entry`, the
    /// state the frame last had in the fold it is in, moved by the tokens the two differ in, or from nothing.
    ///
    /// Each token that `entry` has and `now` does not leaves the list before its rows are taken out, and each
    /// one `now` has and `entry` does not has its rows put in before it joins, as a derive does, so the list
    /// at every touch is the other tokens present. From nothing is the same thing from an empty list, a token
    /// at a time against the ones before it, which is the order [`Accumulator::refresh`] gives: the bits are a
    /// refresh's.
    fn pair_from(
        net: &SymNetwork,
        i: usize,
        folded: [bool; Color::COUNT],
        entry: Option<&FoldEntry>,
        now: &TokenList,
        #[cfg_attr(not(test), allow(unused_variables))] cache: &mut FinnyCache,
    ) -> FoldEntry {
        use super::TOKENS;

        const WORDS: usize = TOKENS.div_ceil(64);
        let mut update = [false; Color::COUNT];
        update[i] = true;
        let mut sums = [vec![0.0; PAIR], vec![0.0; PAIR]];

        // What the boards differ in, through a set of `entry`'s tokens: what is left of it after `now`'s are
        // taken out is what left, and what of `now` was not in it is what came.
        let mut changes = None;
        if let Some(entry) = entry.filter(|e| e.lineage < LINEAGE_LIMIT) {
            let mut had = [0u64; WORDS];
            for &t in entry.tokens.as_slice() {
                had[t as usize / 64] |= 1 << (t % 64);
            }
            let mut came = TokenList::new();
            for &t in now.as_slice() {
                let (word, bit) = (t as usize / 64, 1u64 << (t % 64));
                if had[word] & bit != 0 {
                    had[word] &= !bit;
                } else {
                    came.push(t);
                }
            }
            let mut went = TokenList::new();
            for &t in entry.tokens.as_slice() {
                if had[t as usize / 64] & (1 << (t % 64)) != 0 {
                    went.push(t);
                }
            }
            // A touch reads a row for each other token present, so the moves read about `(went + came) · n`
            // rows and a sum from nothing `n (n − 1) / 2`.
            let n = now.len();
            let moves = went.len() + came.len();
            let lineage = entry.lineage.saturating_add(moves as u32);
            if 2 * moves < n.saturating_sub(1) && lineage <= LINEAGE_LIMIT {
                changes = Some((entry, went, came, lineage));
            }
        }

        let (list, lineage) = match changes {
            Some((entry, went, came, lineage)) => {
                sums[i].copy_from_slice(&entry.pair);
                let mut list = entry.tokens;
                for &t in went.as_slice() {
                    list.remove(t);
                    touch_pairs_hm(&mut sums, net, t, list.as_slice(), -1.0, folded, update);
                }
                for &t in came.as_slice() {
                    touch_pairs_hm(&mut sums, net, t, list.as_slice(), 1.0, folded, update);
                    list.push(t);
                }
                #[cfg(test)]
                {
                    cache.from_entry += 1;
                }
                (list, lineage)
            }
            None => {
                let mut list = TokenList::new();
                for &t in now.as_slice() {
                    touch_pairs_hm(&mut sums, net, t, list.as_slice(), 1.0, folded, update);
                    list.push(t);
                }
                #[cfg(test)]
                {
                    cache.from_nothing += 1;
                }
                (list, 0)
            }
        };
        FoldEntry {
            tokens: list,
            pair: sums[i]
                .as_slice()
                .try_into()
                .expect("a pair sum is PAIR long"),
            lineage,
        }
    }

    /// Frame `i`'s pair, moment and field sums and its tokens, copied from `src`'s.
    ///
    /// The sums are copied at the lengths the shape fixes, and a table the shape does not have is not
    /// touched, so that each copy is a few moves of a known size. Copied at the `Vec`s' own lengths, each is
    /// a call to the C library's `memcpy`, a length of zero included, and in the static build that is
    /// musl's: a few calls a derive came to more of a node's time than the derive's arithmetic.
    fn copy_sums_from(&mut self, src: &Accumulator, i: usize) {
        fn copy<T: Copy, const N: usize>(dst: &mut [T], src: &[T]) {
            let dst: &mut [T; N] = dst.try_into().expect("a sum is the length the shape fixes");
            *dst = *<&[T; N]>::try_from(src).expect("a sum is the length the shape fixes");
        }
        if HAS_PAIR {
            copy::<_, PAIR>(&mut self.pair[i], &src.pair[i]);
        }
        if HAS_MOMENT {
            copy::<_, MOMENT_LEN>(&mut self.moment[i], &src.moment[i]);
        }
        if HAS_FIELD {
            copy::<_, FIELD_LEN>(&mut self.field[i], &src.field[i]);
        }
        self.active[i].clone_from(&src.active[i]);
        self.lineage[i] = src.lineage[i];
    }

    /// The pair, moment and field sums and the sets of tokens of both halves,
    /// taken from what they hold to the position after the move. The power sums
    /// are not touched.
    ///
    /// The two halves name the same pieces in the same order, each as its own
    /// frame numbers them (`tokens::other_frame`), so the pair sums of both are
    /// moved by one call per token, from the black frame's tokens.
    fn touch_tokens(
        &mut self,
        net: &SymNetwork,
        black: &PerspectiveDelta,
        white: &PerspectiveDelta,
    ) {
        let (b, w) = (Color::Black.index(), Color::White.index());
        for (&token, &other) in black.removed().iter().zip(white.removed()) {
            for (i, token) in [(b, token), (w, other)] {
                let at = position_of(&self.active[i], token)
                    .expect("a removed token was not in the accumulator it is removed from");
                self.active[i].swap_remove(at);
            }
            touch_pairs(&mut self.pair, net, token, &self.active[b], -1.0);
            for (i, token) in [(b, token), (w, other)] {
                touch_moment(&mut self.moment[i], net, token, -1.0);
                touch_field(&mut self.field[i], net, token, -1.0);
            }
        }
        for (&token, &other) in black.added().iter().zip(white.added()) {
            touch_pairs(&mut self.pair, net, token, &self.active[b], 1.0);
            for (i, token) in [(b, token), (w, other)] {
                touch_moment(&mut self.moment[i], net, token, 1.0);
                touch_field(&mut self.field[i], net, token, 1.0);
                self.active[i].push(token);
            }
        }
    }
}

/// Where `token` is in `list`, the first place it is: what `iter().position` finds,
/// eight entries at a time. A position has each token once, so there is no second
/// place to be told apart from.
#[cfg(feature = "sym_token")]
#[inline]
fn position_of(list: &[FeatureIndex], token: FeatureIndex) -> Option<usize> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { position_of_avx2(list, token) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        list.iter().position(|&present| present == token)
    }
}

/// [`position_of`] on AVX2: a compare of eight entries and a mask for each whole
/// group, then the entries after the last whole group one at a time.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it,
/// and the build is for the host CPU, so it does.
#[cfg(all(feature = "sym_token", target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn position_of_avx2(list: &[FeatureIndex], token: FeatureIndex) -> Option<usize> {
    use std::arch::x86_64::*;

    let needle = _mm256_set1_epi32(token as i32);
    let mut at = 0;
    while at + 8 <= list.len() {
        // SAFETY: `at + 8 <= list.len()`; the load is unaligned.
        let group = unsafe { _mm256_loadu_si256(list.as_ptr().add(at).cast()) };
        let hits = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(group, needle)));
        if hits != 0 {
            return Some(at + hits.trailing_zeros() as usize);
        }
        at += 8;
    }
    list[at..]
        .iter()
        .position(|&present| present == token)
        .map(|place| at + place)
}

/// What a half's power sums start from when they are built rather than derived.
static ZERO_SUMS: [f32; POWER_LEN] = [0.0; POWER_LEN];

/// Write into `dst` the power sums `src` holds with the powers of the `removed`
/// features' rows taken out and the `added` ones put in, in that order.
///
/// `src` and `dst` are separate, so a derive reads its parent's sums and writes
/// its own in the same pass and never copies the parent's first; a rebuild
/// passes [`ZERO_SUMS`] as `src`.
///
/// The subtraction is not exact: `add` then `sub` of the same row restores the
/// sum only up to f32 rounding. Every shape composes these deltas from the
/// parent's accumulator at every node (a HalfKP half whose king moved starts
/// again from zero), and the search seeds the root with a fresh
/// [`Accumulator::refresh`] at the start of every search, which is what bounds
/// the drift to a search's depth.
///
/// Eight channels at a time, each group's sums read once from `src`, taken
/// through every feature in turn and written once to `dst`, where doing a
/// feature at a time reads and writes all of the sums for each. A channel sees
/// what it always saw — for each feature in order, `sums_k += sign · v^k` and
/// then the power raised by one more multiplication by `v`, `k` from one to
/// `ORDER` — so the sums are the same bits as copying `src` and updating it in
/// place.
fn accumulate(
    src: &[f32],
    dst: &mut [f32],
    embed: &[f32],
    removed: &[FeatureIndex],
    added: &[FeatureIndex],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { accumulate_avx2(src, dst, embed, removed, added) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        accumulate_groups(src, dst, embed, removed, added);
    }
}

/// How many groups of eight channels [`accumulate_avx2`] carries through the
/// features at once: as many as keep their `ORDER` sums, and the row being added, in
/// the sixteen registers, and a whole number of them to a row. Eight for the first
/// order, whose sums are one register a group, and fewer for the higher ones.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const TILE: usize = {
    let groups = DIM / 8;
    let mut tile = if 8 / ORDER > 0 { 8 / ORDER } else { 1 };
    while !groups.is_multiple_of(tile) {
        tile -= 1;
    }
    tile
};

/// [`accumulate_groups`] with a tile of groups taken through the features together,
/// so that a feature's row is found once for the tile where the group at a time
/// finds it again for every group, and the vector instructions are the whole of the
/// loop's body.
///
/// A channel sees what it always saw — for each feature in order, `sums_k += sign ·
/// v^k` and then the power raised by one more multiplication by `v`, `k` from one to
/// `ORDER` — and channels do not read each other, so the sums are the same bits.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it,
/// and the build is for the host CPU, so it does.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(
    src: &[f32],
    dst: &mut [f32],
    embed: &[f32],
    removed: &[FeatureIndex],
    added: &[FeatureIndex],
) {
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    assert_eq!(src.len(), ORDER * DIM);
    assert_eq!(dst.len(), ORDER * DIM);
    // Every row read is `feature · DIM` in, `DIM` long; the check is once per
    // feature and not once per group.
    for &feature in removed.iter().chain(added) {
        assert!(
            (feature as usize + 1) * DIM <= embed.len(),
            "a feature is a row of the embedding"
        );
    }
    for tile in (0..DIM / LANES).step_by(TILE) {
        let at = tile * LANES;
        // `sums[t][k]`: group `t` of the tile, order `k + 1`.
        let mut sums = [[_mm256_setzero_ps(); ORDER]; TILE];
        for (t, group) in sums.iter_mut().enumerate() {
            for (k, slot) in group.iter_mut().enumerate() {
                // SAFETY: `k · DIM + at + t · 8 + 8 <= ORDER · DIM`, which is what
                // `src` holds (asserted above); the load is unaligned.
                *slot = unsafe { _mm256_loadu_ps(src.as_ptr().add(k * DIM + at + t * LANES)) };
            }
        }
        for (features, subtract) in [(removed, true), (added, false)] {
            for &feature in features {
                // SAFETY: the row is `DIM` floats from `feature · DIM`, inside
                // `embed` (asserted above); `at + TILE · 8 <= DIM`.
                let row = unsafe { embed.as_ptr().add(feature as usize * DIM + at) };
                for (t, group) in sums.iter_mut().enumerate() {
                    // SAFETY: as above; the load is unaligned.
                    let value = unsafe { _mm256_loadu_ps(row.add(t * LANES)) };
                    let mut power = value;
                    for (k, slot) in group.iter_mut().enumerate() {
                        *slot = if subtract {
                            _mm256_sub_ps(*slot, power)
                        } else {
                            _mm256_add_ps(*slot, power)
                        };
                        if k + 1 < ORDER {
                            power = _mm256_mul_ps(power, value);
                        }
                    }
                }
            }
        }
        for (t, group) in sums.iter().enumerate() {
            for (k, &slot) in group.iter().enumerate() {
                // SAFETY: as for the loads from `src`, on `dst`.
                unsafe { _mm256_storeu_ps(dst.as_mut_ptr().add(k * DIM + at + t * LANES), slot) };
            }
        }
    }
}

/// [`accumulate`], a group of eight channels at a time; see there. What the AVX2
/// form is held to.
#[allow(dead_code)]
fn accumulate_groups(
    src: &[f32],
    dst: &mut [f32],
    embed: &[f32],
    removed: &[FeatureIndex],
    added: &[FeatureIndex],
) {
    const LANES: usize = 8;
    const _: () = assert!(
        DIM.is_multiple_of(LANES),
        "channels come in whole groups of eight"
    );
    assert_eq!(src.len(), ORDER * DIM);
    assert_eq!(dst.len(), ORDER * DIM);
    let row = |feature: FeatureIndex, at: usize| -> [f32; LANES] {
        let base = feature as usize * DIM + at;
        embed[base..base + LANES]
            .try_into()
            .expect("a group of eight")
    };
    for at in (0..DIM).step_by(LANES) {
        let mut group = [[0.0f32; LANES]; ORDER];
        for (k, sum) in group.iter_mut().enumerate() {
            sum.copy_from_slice(&src[k * DIM + at..k * DIM + at + LANES]);
        }
        for (features, sign) in [(removed, -1.0f32), (added, 1.0f32)] {
            for &feature in features {
                let value = row(feature, at);
                let mut power = value;
                for sum in &mut group {
                    for lane in 0..LANES {
                        sum[lane] += sign * power[lane];
                    }
                    for lane in 0..LANES {
                        power[lane] *= value[lane];
                    }
                }
            }
        }
        for (k, sum) in group.iter().enumerate() {
            dst[k * DIM + at..k * DIM + at + LANES].copy_from_slice(sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One feature at a time over the whole row, with the order as the inner loop:
    /// the form the grouped pass replaced, kept to hold it to the same bits.
    fn one_at_a_time(sums: &mut [f32], embed: &[f32], feature: FeatureIndex, sign: f32) {
        let row = &embed[feature as usize * DIM..(feature as usize + 1) * DIM];
        for (channel, &value) in row.iter().enumerate() {
            let mut power = value;
            for k in 0..ORDER {
                sums[k * DIM + channel] += sign * power;
                power *= value;
            }
        }
    }

    #[cfg(feature = "sym_token")]
    #[test]
    fn the_position_of_a_token_is_where_iter_position_finds_it() {
        let mut state = 3u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as FeatureIndex % 2400
        };
        for len in 0..=48usize {
            for _ in 0..50 {
                let mut list: Vec<FeatureIndex> = Vec::new();
                while list.len() < len {
                    let t = next();
                    if !list.contains(&t) {
                        list.push(t);
                    }
                }
                for token in list.iter().copied().chain([next(), 0, 2399]) {
                    assert_eq!(
                        position_of(&list, token),
                        list.iter().position(|&p| p == token),
                        "len {len} token {token}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_batch_of_features_is_the_features_one_at_a_time_to_the_last_bit() {
        // A table of values of the size the embeddings have, spread either side of
        // zero; a feature is a row of it.
        let tokens = 64;
        let mut state = 11u64;
        let embed: Vec<f32> = (0..tokens * DIM)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 1.2
            })
            .collect();
        let mut pick = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as usize % tokens) as FeatureIndex
        };
        for round in 0..200 {
            let removed: Vec<FeatureIndex> = (0..round % 3).map(|_| pick()).collect();
            let added: Vec<FeatureIndex> = (0..round % 5).map(|_| pick()).collect();
            let start = vec![0.25f32; ORDER * DIM];
            let mut want = start.clone();
            // Whatever the destination held is overwritten, not added to.
            let mut got = vec![f32::NAN; ORDER * DIM];
            for &feature in &removed {
                one_at_a_time(&mut want, &embed, feature, -1.0);
            }
            for &feature in &added {
                one_at_a_time(&mut want, &embed, feature, 1.0);
            }
            accumulate(&start, &mut got, &embed, &removed, &added);
            let mut grouped = vec![f32::NAN; ORDER * DIM];
            accumulate_groups(&start, &mut grouped, &embed, &removed, &added);
            for (at, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "round {round} entry {at}");
                assert_eq!(
                    grouped[at].to_bits(),
                    w.to_bits(),
                    "grouped, round {round} entry {at}"
                );
            }
        }
    }
}
