use std::process::Command;

#[test]
fn executable_rejects_invalid_command_line_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_uav-vision-rs"))
        .args(["--camera-width", "0"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--camera-width"));
}
