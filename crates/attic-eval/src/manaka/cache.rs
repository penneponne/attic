//! The buffers a derive needs, owned across the whole search.

#[cfg(not(feature = "sym_token"))]
use crate::features::{FeatureIndex, MAX_ACTIVE_FEATURES};
#[cfg(feature = "sym_token")]
use crate::tokens::{FeatureIndex, MAX_ACTIVE_FEATURES};

/// Where NNUE keeps its king-bucket refresh cache. Nothing here caches a
/// bucket; what the KP and HalfKP shapes carry instead is the feature list a
/// half is rebuilt through when its own king moves (HalfKP), so that the rebuild
/// does not allocate. It is held here rather than in
/// [`Accumulator`](super::Accumulator) because the accumulator is copied down the
/// tree and this is not — one worker needs one list, not one per ply.
#[cfg(not(feature = "sym_token"))]
#[derive(Debug)]
pub struct FinnyCache {
    pub(super) features: Vec<FeatureIndex>,
}

#[cfg(not(feature = "sym_token"))]
impl Default for FinnyCache {
    fn default() -> Self {
        FinnyCache {
            features: Vec::with_capacity(MAX_ACTIVE_FEATURES),
        }
    }
}

/// Where NNUE keeps its king-bucket refresh cache, and where an `hm` net keeps
/// the state each frame was last in under each fold.
///
/// A frame of an `hm` net whose own king crosses the line between the fifth file
/// and the sixth is read the other way, which changes every board token of it, so
/// its pair sum, over every pair of its forty tokens, is not a step from the
/// parent's. It is a step from the last state the frame had in the fold it now
/// enters, which this keeps: the board then and the board now differ in a dozen
/// tokens or so, and the pair sum moves by their rows alone. The other shapes have
/// no fold and leave this empty; it exists for them because the search allocates
/// one per worker and hands it to every derive.
#[cfg(feature = "sym_token")]
#[derive(Debug, Default)]
pub struct FinnyCache {
    /// `[frame][fold]`: the last state frame `frame` was in while read mirrored
    /// (`true`) or not, or `None` before there was one.
    pub(super) folds: [[Option<FoldEntry>; 2]; 2],
    /// Which network the entries are sums of: its address and its embedding's, as
    /// NNUE's cache records them, so that a network loaded in its place empties
    /// the cache rather than being read through another's sums.
    pub(super) net: Option<(usize, usize)>,
    /// How many crossings were built from an entry, and how many from nothing: the
    /// tests' evidence that the first kind happened.
    #[cfg(test)]
    pub(super) from_entry: usize,
    #[cfg(test)]
    pub(super) from_nothing: usize,
}

/// One frame's state in one fold, as [`FinnyCache`] keeps it.
#[cfg(feature = "sym_token")]
#[derive(Clone, Copy, Debug)]
pub(super) struct FoldEntry {
    /// The frame's tokens, unreflected and numbered as the black frame numbers
    /// them, which is how the pair table is read.
    pub(super) tokens: TokenList,
    /// The frame's pair sum over those tokens, in that fold.
    pub(super) pair: [super::accumulator::PairSum; super::PAIR],
    /// Touches the sum has taken since it was last built from nothing; see
    /// `Accumulator::lineage`.
    pub(super) lineage: u32,
}

/// A frame's tokens in a fixed array, so that keeping one allocates nothing.
#[cfg(feature = "sym_token")]
#[derive(Clone, Copy, Debug)]
pub(super) struct TokenList {
    at: [FeatureIndex; MAX_ACTIVE_FEATURES],
    len: usize,
}

#[cfg(feature = "sym_token")]
impl TokenList {
    pub(super) const fn new() -> TokenList {
        TokenList {
            at: [0; MAX_ACTIVE_FEATURES],
            len: 0,
        }
    }

    /// `tokens`, numbered as frame `frame` numbers them, turned to the black
    /// frame's numbering (`other_frame` when `frame` is the white).
    pub(super) fn black(tokens: &[FeatureIndex], frame: usize) -> TokenList {
        let mut list = TokenList::new();
        for &token in tokens {
            list.push(match frame {
                0 => token,
                _ => crate::tokens::other_frame(token),
            });
        }
        list
    }

    pub(super) fn as_slice(&self) -> &[FeatureIndex] {
        &self.at[..self.len]
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// # Panics
    /// Panics if the list already holds [`MAX_ACTIVE_FEATURES`] tokens.
    pub(super) fn push(&mut self, token: FeatureIndex) {
        self.at[self.len] = token;
        self.len += 1;
    }

    /// Take `token` out, moving the last one into its place.
    ///
    /// # Panics
    /// Panics if `token` is not in the list.
    pub(super) fn remove(&mut self, token: FeatureIndex) {
        let at = self
            .as_slice()
            .iter()
            .position(|&t| t == token)
            .expect("a token taken out of a list is in it");
        self.len -= 1;
        self.at[at] = self.at[self.len];
    }
}

impl FinnyCache {
    pub fn new() -> Box<Self> {
        Box::default()
    }
}

#[cfg(feature = "sym_token")]
impl FinnyCache {
    /// Drop every entry if they are sums of another network than `net`.
    pub(super) fn for_net(&mut self, net: &super::SymNetwork) {
        let token = (
            net as *const super::SymNetwork as usize,
            net.embed.as_ptr() as usize,
        );
        if self.net != Some(token) {
            self.folds = Default::default();
            self.net = Some(token);
        }
    }
}
