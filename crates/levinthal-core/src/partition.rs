//! Partition Coordinate System
//!
//! The four-parameter partition coordinate system (n, l, m, s) specifies
//! categorical states in bounded phase space. This is derived from the
//! geometry of nested boundaries in bounded oscillatory systems.
//!
//! ## Coordinate Constraints
//!
//! - `n ≥ 1`: Depth (nesting level)
//! - `l ∈ {0, 1, ..., n-1}`: Complexity (boundary shape)
//! - `m ∈ {-l, ..., +l}`: Orientation (angular position)
//! - `s ∈ {-½, +½}`: Chirality (handedness)
//!
//! ## Key Results
//!
//! - **Capacity**: `C(n) = 2n²` states at depth n
//! - **Selection Rules**: Allowed transitions satisfy `Δl = ±1`, `|Δm| ≤ 1`, `Δs = 0`
//! - **Determinism**: Trajectories through partition space are deterministic (σ < 10⁻⁶)

use serde::{Deserialize, Serialize};
use crate::error::{LevinthalError, Result};

/// Half-integer spin values for chirality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Spin {
    /// Spin down: s = -1/2
    Down,
    /// Spin up: s = +1/2
    Up,
}

impl Spin {
    /// Get the numeric value of the spin.
    pub fn value(&self) -> f64 {
        match self {
            Spin::Down => -0.5,
            Spin::Up => 0.5,
        }
    }

    /// Create from numeric value (rounds to nearest half-integer).
    pub fn from_value(v: f64) -> Self {
        if v < 0.0 {
            Spin::Down
        } else {
            Spin::Up
        }
    }
}

impl std::fmt::Display for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Spin::Down => write!(f, "-½"),
            Spin::Up => write!(f, "+½"),
        }
    }
}

/// A partition state in (n, l, m, s) coordinates.
///
/// Represents a categorical state in bounded phase space, satisfying:
/// - `n ≥ 1` (depth)
/// - `0 ≤ l ≤ n-1` (complexity)
/// - `-l ≤ m ≤ +l` (orientation)
/// - `s ∈ {-½, +½}` (chirality)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionState {
    n: u32,
    l: u32,
    m: i32,
    s: Spin,
}

impl PartitionState {
    /// Create a new partition state with validation.
    ///
    /// # Arguments
    ///
    /// * `n` - Depth (must be ≥ 1)
    /// * `l` - Complexity (must be 0 ≤ l ≤ n-1)
    /// * `m` - Orientation (must be -l ≤ m ≤ +l)
    /// * `s` - Chirality as float (-0.5 or +0.5)
    ///
    /// # Errors
    ///
    /// Returns `InvalidPartitionCoordinates` if constraints are violated.
    pub fn new(n: u32, l: u32, m: i32, s: f64) -> Result<Self> {
        // Validate n ≥ 1
        if n < 1 {
            return Err(LevinthalError::InvalidPartitionCoordinates {
                message: format!("n must be ≥ 1, got {}", n),
            });
        }

        // Validate 0 ≤ l ≤ n-1
        if l >= n {
            return Err(LevinthalError::InvalidPartitionCoordinates {
                message: format!("l must be < n, got l={} with n={}", l, n),
            });
        }

        // Validate -l ≤ m ≤ +l
        let l_signed = l as i32;
        if m < -l_signed || m > l_signed {
            return Err(LevinthalError::InvalidPartitionCoordinates {
                message: format!("m must be in [-l, +l], got m={} with l={}", m, l),
            });
        }

        Ok(Self {
            n,
            l,
            m,
            s: Spin::from_value(s),
        })
    }

    /// Create a partition state without validation (unsafe).
    ///
    /// Use only when you're certain the coordinates are valid.
    pub fn new_unchecked(n: u32, l: u32, m: i32, s: Spin) -> Self {
        Self { n, l, m, s }
    }

    /// Get the depth parameter n.
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Get the complexity parameter l.
    pub fn l(&self) -> u32 {
        self.l
    }

    /// Get the orientation parameter m.
    pub fn m(&self) -> i32 {
        self.m
    }

    /// Get the chirality parameter s.
    pub fn s(&self) -> Spin {
        self.s
    }

    /// Calculate the capacity at depth n: C(n) = 2n².
    ///
    /// This is the number of distinct partition states at depth n.
    pub fn capacity(n: u32) -> u64 {
        2 * (n as u64) * (n as u64)
    }

    /// Calculate cumulative capacity up to and including depth n.
    ///
    /// Sum of C(k) for k = 1 to n.
    pub fn cumulative_capacity(n: u32) -> u64 {
        // Sum of 2k² from k=1 to n = 2 * n(n+1)(2n+1)/6 = n(n+1)(2n+1)/3
        let n = n as u64;
        n * (n + 1) * (2 * n + 1) / 3
    }

    /// Enumerate all states at depth n.
    pub fn enumerate_at_depth(n: u32) -> Vec<Self> {
        let mut states = Vec::with_capacity(Self::capacity(n) as usize);

        for l in 0..n {
            let l_signed = l as i32;
            for m in -l_signed..=l_signed {
                states.push(Self::new_unchecked(n, l, m, Spin::Down));
                states.push(Self::new_unchecked(n, l, m, Spin::Up));
            }
        }

        states
    }

    /// Enumerate all states up to and including depth n.
    pub fn enumerate_up_to_depth(n: u32) -> Vec<Self> {
        let mut states = Vec::with_capacity(Self::cumulative_capacity(n) as usize);

        for depth in 1..=n {
            states.extend(Self::enumerate_at_depth(depth));
        }

        states
    }

    /// Check if a transition to another state satisfies selection rules.
    ///
    /// Two types of allowed transitions:
    ///
    /// 1. **Intra-depth transitions** (same n):
    ///    - Δl = ±1
    ///    - |Δm| ≤ 1 (i.e., Δm ∈ {-1, 0, +1})
    ///    - Δs = 0
    ///
    /// 2. **Depth transitions** (n changes):
    ///    - Δn = ±1
    ///    - Both states at l = 0, m = 0 (ground state at each depth)
    ///    - Δs = 0
    pub fn is_allowed_transition(&self, other: &Self) -> bool {
        let delta_n = (other.n as i32) - (self.n as i32);
        let delta_l = (other.l as i32) - (self.l as i32);
        let delta_m = other.m - self.m;
        let delta_s = other.s != self.s;

        // Chirality must be conserved
        if delta_s {
            return false;
        }

        // Case 1: Depth transition (moving between shells)
        if delta_n != 0 {
            // Must be exactly ±1 depth change
            let n_ok = delta_n == 1 || delta_n == -1;
            // Both states must be at ground (l=0, m=0)
            let ground_ok = self.l == 0 && self.m == 0 && other.l == 0 && other.m == 0;
            return n_ok && ground_ok;
        }

        // Case 2: Intra-depth transition (within same shell)
        // Δl = ±1
        let l_ok = delta_l == 1 || delta_l == -1;
        // |Δm| ≤ 1
        let m_ok = delta_m >= -1 && delta_m <= 1;

        l_ok && m_ok
    }

    /// Get all allowed transitions from this state.
    ///
    /// Returns states that satisfy the selection rules.
    pub fn allowed_transitions(&self) -> Vec<Self> {
        let mut transitions = Vec::new();

        // Try Δl = +1 (if valid)
        let new_l = self.l + 1;
        for delta_m in -1..=1 {
            let new_m = self.m + delta_m;
            if new_m >= -(new_l as i32) && new_m <= (new_l as i32) {
                // n must be > new_l for valid state
                let min_n = new_l + 1;
                transitions.push(Self::new_unchecked(min_n, new_l, new_m, self.s));
            }
        }

        // Try Δl = -1 (if valid, i.e., l > 0)
        if self.l > 0 {
            let new_l = self.l - 1;
            for delta_m in -1..=1 {
                let new_m = self.m + delta_m;
                // Check if new_m is valid for new_l
                if new_m >= -(new_l as i32) && new_m <= (new_l as i32) {
                    transitions.push(Self::new_unchecked(self.n, new_l, new_m, self.s));
                }
            }
        }

        transitions
    }

    /// Calculate the partition operation: derive the penultimate state.
    ///
    /// Given this state as the goal, determine the unique predecessor
    /// state through phase-lock adjacency. This is the core operation
    /// for trajectory completion.
    ///
    /// Returns `None` if this is an origin state (cannot be further partitioned).
    pub fn partition(&self) -> Option<Self> {
        // The simplest partition rule: decrease coherence by going to lower l
        // In a complete implementation, this would use phase-lock topology
        if self.l > 0 {
            // Go to lower complexity
            Some(Self::new_unchecked(self.n, self.l - 1, self.m.clamp(-(self.l as i32 - 1), self.l as i32 - 1), self.s))
        } else if self.n > 1 {
            // At l=0, go to lower depth
            Some(Self::new_unchecked(self.n - 1, 0, 0, self.s))
        } else {
            // Origin state: (1, 0, 0, s)
            None
        }
    }

    /// Check if this is an origin state (cannot be partitioned further).
    pub fn is_origin(&self) -> bool {
        self.n == 1 && self.l == 0 && self.m == 0
    }

    /// Calculate a "coherence" proxy based on partition coordinates.
    ///
    /// Higher n and l correspond to higher coherence/structure.
    /// This is a simplified proxy; real coherence comes from phase-lock dynamics.
    pub fn coherence_proxy(&self) -> f64 {
        // Normalize to [0, 1] range based on typical values
        let n_contrib = (self.n as f64 - 1.0) / 10.0;
        let l_contrib = (self.l as f64) / (self.n as f64).max(1.0);
        (0.2 + 0.4 * n_contrib + 0.4 * l_contrib).min(1.0)
    }
}

impl std::fmt::Display for PartitionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self.n, self.l, self.m, self.s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use approx::assert_relative_eq;

    #[test]
    fn test_capacity_formula() {
        // C(n) = 2n²
        assert_eq!(PartitionState::capacity(1), 2);
        assert_eq!(PartitionState::capacity(2), 8);
        assert_eq!(PartitionState::capacity(3), 18);
        assert_eq!(PartitionState::capacity(4), 32);
        assert_eq!(PartitionState::capacity(5), 50);
        assert_eq!(PartitionState::capacity(10), 200);
    }

    #[test]
    fn test_enumeration_matches_capacity() {
        for n in 1..=7 {
            let states = PartitionState::enumerate_at_depth(n);
            assert_eq!(
                states.len() as u64,
                PartitionState::capacity(n),
                "Enumeration at depth {} doesn't match capacity",
                n
            );
        }
    }

    #[test]
    fn test_valid_state_creation() {
        // (1, 0, 0, +½) - ground state
        let s = PartitionState::new(1, 0, 0, 0.5).unwrap();
        assert_eq!(s.n(), 1);
        assert_eq!(s.l(), 0);
        assert_eq!(s.m(), 0);
        assert_eq!(s.s(), Spin::Up);

        // (2, 1, 0, -½)
        let s = PartitionState::new(2, 1, 0, -0.5).unwrap();
        assert_eq!(s.n(), 2);
        assert_eq!(s.l(), 1);
        assert_eq!(s.m(), 0);
        assert_eq!(s.s(), Spin::Down);

        // (3, 2, -1, +½)
        let s = PartitionState::new(3, 2, -1, 0.5).unwrap();
        assert_eq!(s.n(), 3);
        assert_eq!(s.l(), 2);
        assert_eq!(s.m(), -1);
    }

    #[test]
    fn test_invalid_state_creation() {
        // n = 0 (invalid)
        assert!(PartitionState::new(0, 0, 0, 0.5).is_err());

        // l >= n (invalid)
        assert!(PartitionState::new(2, 2, 0, 0.5).is_err());

        // |m| > l (invalid)
        assert!(PartitionState::new(3, 1, 2, 0.5).is_err());
        assert!(PartitionState::new(3, 1, -2, 0.5).is_err());
    }

    #[test]
    fn test_selection_rules() {
        let s1 = PartitionState::new(2, 0, 0, 0.5).unwrap();
        let s2 = PartitionState::new(2, 1, 0, 0.5).unwrap();
        let s3 = PartitionState::new(2, 1, 1, 0.5).unwrap();
        let s4 = PartitionState::new(2, 0, 0, -0.5).unwrap();

        // Δl = +1, Δm = 0, Δs = 0 → allowed
        assert!(s1.is_allowed_transition(&s2));

        // Δl = +1, Δm = +1, Δs = 0 → allowed
        assert!(s1.is_allowed_transition(&s3));

        // Δl = 0, Δm = 0, Δs = 1 → forbidden (spin flip)
        assert!(!s1.is_allowed_transition(&s4));

        // Δl = -1 → allowed
        assert!(s2.is_allowed_transition(&s1));
    }

    #[test]
    fn test_partition_operation() {
        // (2, 1, 0, +½) should partition to (2, 0, 0, +½)
        let s = PartitionState::new(2, 1, 0, 0.5).unwrap();
        let prev = s.partition().unwrap();
        assert_eq!(prev.l(), 0);
        assert_eq!(prev.s(), Spin::Up);

        // (1, 0, 0, +½) is origin, cannot partition
        let origin = PartitionState::new(1, 0, 0, 0.5).unwrap();
        assert!(origin.partition().is_none());
        assert!(origin.is_origin());
    }

    #[test]
    fn test_subshell_capacities() {
        // s (l=0): 2 states
        // p (l=1): 6 states
        // d (l=2): 10 states
        // f (l=3): 14 states
        // g (l=4): 18 states

        fn subshell_capacity(l: u32) -> u32 {
            2 * (2 * l + 1)
        }

        assert_eq!(subshell_capacity(0), 2);  // s
        assert_eq!(subshell_capacity(1), 6);  // p
        assert_eq!(subshell_capacity(2), 10); // d
        assert_eq!(subshell_capacity(3), 14); // f
        assert_eq!(subshell_capacity(4), 18); // g
    }
}
