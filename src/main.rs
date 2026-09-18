// Responsibility: Start the executable, pass command-line arguments to the library, and map the final result to the process exit status.
// Input: Command-line arguments and process environment provided by the operating system.
// Output: A running UAV vision application or a user-facing startup error and non-zero exit status.

fn main() {
    if let Err(error) = uav_vision_rs::config::cli::parse() {
        error.exit();
    }
}
