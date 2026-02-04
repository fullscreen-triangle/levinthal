//! Levinthal CLI
//!
//! Command-line interface for the Levinthal protein folding framework.

use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(name = "levinthal")]
#[command(author, version, about = "Levinthal protein folding framework", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a protein sequence
    Analyze {
        /// Protein sequence (single-letter codes)
        #[arg(short, long)]
        sequence: String,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Calculate partition states for a sequence
    Partition {
        /// Protein sequence
        #[arg(short, long)]
        sequence: String,

        /// Maximum depth to enumerate
        #[arg(short, long, default_value = "3")]
        depth: u32,
    },

    /// Complete a trajectory from a goal state
    Complete {
        /// Goal state as (n, l, m, s)
        #[arg(short = 'n', long)]
        depth: u32,

        #[arg(short = 'l', long)]
        complexity: u32,

        #[arg(short = 'm', long, default_value = "0")]
        orientation: i32,

        #[arg(short = 's', long, default_value = "0.5")]
        spin: f64,
    },

    /// Simulate protein folding with Kuramoto dynamics
    Fold {
        /// Number of residues
        #[arg(short, long)]
        residues: usize,

        /// Coupling strength
        #[arg(short, long, default_value = "1.0")]
        coupling: f64,

        /// Maximum simulation steps
        #[arg(long, default_value = "10000")]
        max_steps: usize,

        /// Time step
        #[arg(long, default_value = "0.01")]
        dt: f64,
    },

    /// Analyze MS/MS spectrum
    Msms {
        /// Input spectrum file (mzML or MGF format)
        #[arg(short, long)]
        input: String,

        /// Output file
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Convert ternary string to coordinates
    Ternary {
        /// Ternary string (e.g., "012102")
        #[arg(short, long)]
        string: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze { sequence, format } => {
            commands::analyze(&sequence, &format);
        }
        Commands::Partition { sequence, depth } => {
            commands::partition(&sequence, depth);
        }
        Commands::Complete { depth, complexity, orientation, spin } => {
            commands::complete(depth, complexity, orientation, spin);
        }
        Commands::Fold { residues, coupling, max_steps, dt } => {
            commands::fold(residues, coupling, max_steps, dt);
        }
        Commands::Msms { input, output } => {
            commands::msms(&input, output.as_deref());
        }
        Commands::Ternary { string } => {
            commands::ternary(&string);
        }
    }
}
