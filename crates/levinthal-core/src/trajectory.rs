//! Trajectory Module
//!
//! Implements trajectories through partition space - ordered sequences of states
//! that represent protein folding pathways.
//!
//! ## Core Insight
//!
//! Protein folding is not forward search through conformational space but
//! **backward derivation** through categorical space. The native structure
//! determines its own folding pathway through the geometry of partition
//! coordinates.
//!
//! ## Trajectory Completion
//!
//! Given a goal state (native structure), the trajectory is derived by
//! iteratively applying the partition operation until reaching the origin.
//! This converts O(3^N) forward search into O(log₃ N) backward derivation.
//!
//! ## Selection Rules
//!
//! Valid trajectories satisfy selection rules at each step:
//! - Δl = ±1 (complexity changes by exactly one)
//! - |Δm| ≤ 1 (orientation changes by at most one)
//! - Δs = 0 (chirality is conserved)

use serde::{Deserialize, Serialize};
use crate::error::{LevinthalError, Result};
use crate::partition::PartitionState;
use crate::sentropy::SEntropyCoord;
use crate::ternary::TernaryString;

/// A trajectory through partition space.
///
/// An ordered sequence of partition states representing a folding pathway.
/// Trajectories are typically derived backward from the goal state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trajectory {
    /// The ordered sequence of states (from origin to goal).
    states: Vec<PartitionState>,
}

impl Trajectory {
    /// Create a new trajectory from a sequence of states.
    ///
    /// The states should be ordered from origin to goal.
    pub fn new(states: Vec<PartitionState>) -> Result<Self> {
        if states.is_empty() {
            return Err(LevinthalError::EmptyTrajectory);
        }

        let traj = Self { states };
        traj.validate()?;
        Ok(traj)
    }

    /// Create a trajectory without validation.
    ///
    /// Use only when you're certain the trajectory is valid.
    pub fn new_unchecked(states: Vec<PartitionState>) -> Self {
        Self { states }
    }

    /// Complete a trajectory from a goal state back to the origin.
    ///
    /// This is the core operation: backward derivation through partition space.
    /// The trajectory is built by iteratively applying the partition operation
    /// until reaching an origin state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use levinthal_core::partition::PartitionState;
    /// use levinthal_core::trajectory::Trajectory;
    ///
    /// let goal = PartitionState::new(3, 2, 1, 0.5).unwrap();
    /// let traj = Trajectory::complete(goal);
    /// assert!(traj.origin().is_origin());
    /// ```
    pub fn complete(goal: PartitionState) -> Self {
        let mut states = vec![goal];
        let mut current = goal;

        // Backward derivation: iterate partition operation until origin
        while let Some(predecessor) = current.partition() {
            states.push(predecessor);
            current = predecessor;
        }

        // Reverse to get origin-to-goal order
        states.reverse();
        Self { states }
    }

    /// Get the number of states in the trajectory.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Get the origin state (first state).
    pub fn origin(&self) -> &PartitionState {
        self.states.first().expect("Trajectory cannot be empty")
    }

    /// Get the goal state (final state).
    pub fn goal(&self) -> &PartitionState {
        self.states.last().expect("Trajectory cannot be empty")
    }

    /// Get all states in the trajectory.
    pub fn states(&self) -> &[PartitionState] {
        &self.states
    }

    /// Get a state at a specific index.
    pub fn get(&self, index: usize) -> Option<&PartitionState> {
        self.states.get(index)
    }

    /// Iterate over states in the trajectory.
    pub fn iter(&self) -> impl Iterator<Item = &PartitionState> {
        self.states.iter()
    }

    /// Validate that the trajectory satisfies selection rules.
    ///
    /// Checks that each consecutive pair of states represents an allowed
    /// transition according to the selection rules.
    pub fn validate(&self) -> Result<()> {
        if self.states.is_empty() {
            return Err(LevinthalError::EmptyTrajectory);
        }

        for (i, window) in self.states.windows(2).enumerate() {
            let (from, to) = (&window[0], &window[1]);
            if !from.is_allowed_transition(to) {
                return Err(LevinthalError::TrajectoryDiscontinuity { index: i });
            }
        }

        Ok(())
    }

    /// Check if the trajectory is continuous (all transitions are allowed).
    pub fn is_continuous(&self) -> bool {
        self.validate().is_ok()
    }

    /// Calculate the total complexity change Σ|Δl|.
    pub fn total_complexity_change(&self) -> u32 {
        self.states
            .windows(2)
            .map(|w| (w[1].l() as i32 - w[0].l() as i32).unsigned_abs())
            .sum()
    }

    /// Calculate the maximum complexity reached.
    pub fn max_complexity(&self) -> u32 {
        self.states.iter().map(|s| s.l()).max().unwrap_or(0)
    }

    /// Calculate the maximum depth reached.
    pub fn max_depth(&self) -> u32 {
        self.states.iter().map(|s| s.n()).max().unwrap_or(1)
    }

    /// Get the coherence profile along the trajectory.
    ///
    /// Returns the coherence proxy for each state.
    pub fn coherence_profile(&self) -> Vec<f64> {
        self.states.iter().map(|s| s.coherence_proxy()).collect()
    }

    /// Calculate the average coherence along the trajectory.
    pub fn average_coherence(&self) -> f64 {
        let profile = self.coherence_profile();
        if profile.is_empty() {
            return 0.0;
        }
        profile.iter().sum::<f64>() / profile.len() as f64
    }

    /// Extend the trajectory with additional states.
    ///
    /// Validates that the transition from the current goal to the new states
    /// follows selection rules.
    pub fn extend(&mut self, states: &[PartitionState]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }

        // Validate transition from current goal to first new state
        let current_goal = self.goal();
        if !current_goal.is_allowed_transition(&states[0]) {
            return Err(LevinthalError::TrajectoryDiscontinuity {
                index: self.states.len() - 1,
            });
        }

        // Validate internal transitions in new states
        for (i, window) in states.windows(2).enumerate() {
            if !window[0].is_allowed_transition(&window[1]) {
                return Err(LevinthalError::TrajectoryDiscontinuity {
                    index: self.states.len() + i,
                });
            }
        }

        self.states.extend_from_slice(states);
        Ok(())
    }

    /// Create a sub-trajectory from index `start` to `end`.
    pub fn slice(&self, start: usize, end: usize) -> Option<Self> {
        if start >= end || end > self.states.len() {
            return None;
        }
        Some(Self {
            states: self.states[start..end].to_vec(),
        })
    }

    /// Reverse the trajectory (goal becomes origin).
    pub fn reverse(&self) -> Self {
        let mut states = self.states.clone();
        states.reverse();
        Self { states }
    }

    /// Compose two trajectories (this followed by other).
    ///
    /// The goal of `self` must match the origin of `other`.
    pub fn compose(&self, other: &Self) -> Result<Self> {
        // Check if trajectories can be composed
        if self.goal() != other.origin() {
            return Err(LevinthalError::TrajectoryDiscontinuity {
                index: self.states.len() - 1,
            });
        }

        let mut states = self.states.clone();
        // Skip the first state of other (it's the same as our goal)
        states.extend_from_slice(&other.states[1..]);
        Ok(Self { states })
    }
}

impl std::fmt::Display for Trajectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Trajectory[")?;
        for (i, state) in self.states.iter().enumerate() {
            if i > 0 {
                write!(f, " → ")?;
            }
            write!(f, "{}", state)?;
        }
        write!(f, "]")
    }
}

impl IntoIterator for Trajectory {
    type Item = PartitionState;
    type IntoIter = std::vec::IntoIter<PartitionState>;

    fn into_iter(self) -> Self::IntoIter {
        self.states.into_iter()
    }
}

impl<'a> IntoIterator for &'a Trajectory {
    type Item = &'a PartitionState;
    type IntoIter = std::slice::Iter<'a, PartitionState>;

    fn into_iter(self) -> Self::IntoIter {
        self.states.iter()
    }
}

/// A trajectory annotated with S-entropy coordinates.
///
/// Associates each partition state with its corresponding S-entropy position,
/// enabling visualization and analysis in continuous space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedTrajectory {
    /// The underlying partition trajectory.
    trajectory: Trajectory,
    /// S-entropy coordinates for each state.
    sentropy_coords: Vec<SEntropyCoord>,
    /// Optional ternary encoding.
    ternary: Option<TernaryString>,
}

impl AnnotatedTrajectory {
    /// Create an annotated trajectory with S-entropy coordinates.
    pub fn new(trajectory: Trajectory, sentropy_coords: Vec<SEntropyCoord>) -> Result<Self> {
        if trajectory.len() != sentropy_coords.len() {
            return Err(LevinthalError::TrajectoryDiscontinuity {
                index: trajectory.len(),
            });
        }
        Ok(Self {
            trajectory,
            sentropy_coords,
            ternary: None,
        })
    }

    /// Create with ternary encoding.
    pub fn with_ternary(mut self, ternary: TernaryString) -> Self {
        self.ternary = Some(ternary);
        self
    }

    /// Get the underlying trajectory.
    pub fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    /// Get S-entropy coordinates.
    pub fn sentropy_coords(&self) -> &[SEntropyCoord] {
        &self.sentropy_coords
    }

    /// Get ternary encoding if present.
    pub fn ternary(&self) -> Option<&TernaryString> {
        self.ternary.as_ref()
    }

    /// Calculate total path length in S-entropy space.
    pub fn path_length(&self) -> f64 {
        self.sentropy_coords
            .windows(2)
            .map(|w| w[0].distance(&w[1]))
            .sum()
    }

    /// Calculate the direct distance from origin to goal in S-entropy space.
    pub fn direct_distance(&self) -> f64 {
        if self.sentropy_coords.len() < 2 {
            return 0.0;
        }
        let first = &self.sentropy_coords[0];
        let last = self.sentropy_coords.last().unwrap();
        first.distance(last)
    }

    /// Calculate the path efficiency (direct_distance / path_length).
    ///
    /// A value of 1.0 means the path is perfectly straight.
    /// Lower values indicate more tortuous paths.
    pub fn efficiency(&self) -> f64 {
        let path_len = self.path_length();
        if path_len < 1e-10 {
            return 1.0;
        }
        self.direct_distance() / path_len
    }
}

/// Builder for constructing trajectories step by step.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryBuilder {
    states: Vec<PartitionState>,
}

impl TrajectoryBuilder {
    /// Create a new trajectory builder.
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Start from an origin state.
    pub fn from_origin(origin: PartitionState) -> Self {
        Self {
            states: vec![origin],
        }
    }

    /// Add a state to the trajectory.
    ///
    /// Validates that the transition is allowed if there's a previous state.
    pub fn push(&mut self, state: PartitionState) -> Result<&mut Self> {
        if let Some(last) = self.states.last() {
            if !last.is_allowed_transition(&state) {
                return Err(LevinthalError::TrajectoryDiscontinuity {
                    index: self.states.len() - 1,
                });
            }
        }
        self.states.push(state);
        Ok(self)
    }

    /// Add a state without validation.
    pub fn push_unchecked(&mut self, state: PartitionState) -> &mut Self {
        self.states.push(state);
        self
    }

    /// Build the trajectory.
    pub fn build(self) -> Result<Trajectory> {
        Trajectory::new(self.states)
    }

    /// Build without final validation.
    pub fn build_unchecked(self) -> Trajectory {
        Trajectory::new_unchecked(self.states)
    }

    /// Get the current number of states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_complete() {
        // Complete a trajectory from (3, 2, 1, +½)
        let goal = PartitionState::new(3, 2, 1, 0.5).unwrap();
        let traj = Trajectory::complete(goal);

        // Should start from an origin state
        assert!(traj.origin().is_origin());

        // Should end at the goal
        assert_eq!(*traj.goal(), goal);

        // Should be continuous
        assert!(traj.is_continuous());
    }

    #[test]
    fn test_trajectory_from_origin() {
        // (1, 0, 0, +½) is an origin state
        let origin = PartitionState::new(1, 0, 0, 0.5).unwrap();
        let traj = Trajectory::complete(origin);

        // Should have exactly one state
        assert_eq!(traj.len(), 1);
        assert!(traj.origin().is_origin());
    }

    #[test]
    fn test_trajectory_validation() {
        // Valid trajectory
        let s1 = PartitionState::new(2, 0, 0, 0.5).unwrap();
        let s2 = PartitionState::new(2, 1, 0, 0.5).unwrap();
        let traj = Trajectory::new(vec![s1, s2]).unwrap();
        assert!(traj.is_continuous());

        // Invalid trajectory (Δl = 2)
        let s3 = PartitionState::new(3, 2, 0, 0.5).unwrap();
        assert!(Trajectory::new(vec![s1, s3]).is_err());
    }

    #[test]
    fn test_trajectory_builder() {
        let mut builder = TrajectoryBuilder::new();

        let s1 = PartitionState::new(2, 0, 0, 0.5).unwrap();
        let s2 = PartitionState::new(2, 1, 0, 0.5).unwrap();

        builder.push(s1).unwrap();
        builder.push(s2).unwrap();

        let traj = builder.build().unwrap();
        assert_eq!(traj.len(), 2);
        assert!(traj.is_continuous());
    }

    #[test]
    fn test_trajectory_builder_invalid() {
        let mut builder = TrajectoryBuilder::new();

        let s1 = PartitionState::new(2, 0, 0, 0.5).unwrap();
        let s2 = PartitionState::new(3, 2, 0, 0.5).unwrap(); // Invalid: Δl = 2

        builder.push(s1).unwrap();
        assert!(builder.push(s2).is_err());
    }

    #[test]
    fn test_complexity_metrics() {
        let goal = PartitionState::new(3, 2, 0, 0.5).unwrap();
        let traj = Trajectory::complete(goal);

        assert!(traj.max_complexity() >= 2);
        assert!(traj.total_complexity_change() >= 2);
    }

    #[test]
    fn test_coherence_profile() {
        let goal = PartitionState::new(3, 2, 1, 0.5).unwrap();
        let traj = Trajectory::complete(goal);

        let profile = traj.coherence_profile();
        assert_eq!(profile.len(), traj.len());

        // All coherence values should be in [0, 1]
        for c in profile {
            assert!(c >= 0.0 && c <= 1.0);
        }
    }

    #[test]
    fn test_trajectory_reverse() {
        let goal = PartitionState::new(2, 1, 0, 0.5).unwrap();
        let traj = Trajectory::complete(goal);
        let reversed = traj.reverse();

        assert_eq!(traj.origin(), reversed.goal());
        assert_eq!(traj.goal(), reversed.origin());
    }

    #[test]
    fn test_trajectory_slice() {
        let goal = PartitionState::new(3, 2, 0, 0.5).unwrap();
        let traj = Trajectory::complete(goal);

        if traj.len() >= 2 {
            let slice = traj.slice(0, 2).unwrap();
            assert_eq!(slice.len(), 2);
        }
    }

    #[test]
    fn test_empty_trajectory_fails() {
        assert!(Trajectory::new(vec![]).is_err());
    }
}
