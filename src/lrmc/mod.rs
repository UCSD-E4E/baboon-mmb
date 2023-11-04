pub fn lrmc(opt: &crate::Opt) {
    let video_file = &opt.video_file;
    let mut cap = opencv::videoio::VideoCapture::from_file(video_file, 0).unwrap();

    if !opencv::videoio::VideoCaptureTraitConst::is_opened(&cap).unwrap() {
        eprintln!("Unable to open video: {}", video_file);
        std::process::exit(1);
    }

    let fps: i32 = opencv::videoio::VideoCaptureTraitConst::get(&cap, opencv::videoio::CAP_PROP_FPS)
        .unwrap() as i32;
    let frame_count =
        opencv::videoio::VideoCaptureTraitConst::get(&cap, opencv::videoio::CAP_PROP_FRAME_COUNT)
            .unwrap() as i32;

    let n = (frame_count / (opt.l * fps)) as i32;
    let lrmc_script = format!(
        "addpath('src/lrmc'); fRMC({}, {}, {}, {}, {}, {}); exit",
        opt.max_niter_param, opt.gamma1_param, opt.gamma2_param, n, frame_count, opt.kernel
    );

    let status = std::process::Command::new("matlab")
        .arg("-nodisplay")
        .arg("-nosplash")
        .arg("-nodesktop")
        .arg("-r")
        .arg(&lrmc_script)
        //.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("Failed to execute process");

    if !status.success() {
        eprintln!("Failed to execute process");
        std::process::exit(1);
    }

    opencv::videoio::VideoCaptureTrait::release(&mut cap).unwrap();
}
