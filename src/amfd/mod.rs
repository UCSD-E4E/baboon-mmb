pub fn amfd(opt: &crate::Opt) {
    let video_file = &opt.video_file;
    let mut cap = opencv::videoio::VideoCapture::from_file(video_file, 0).unwrap();

    if !opencv::videoio::VideoCaptureTraitConst::is_opened(&cap).unwrap() {
        eprintln!("Unable to open video: {}", video_file);
        std::process::exit(1);
    }

    // We want to know the height and width of the video frames in case we need to create a black image of the same size.
    let height =
        opencv::videoio::VideoCaptureTraitConst::get(&cap, opencv::videoio::CAP_PROP_FRAME_HEIGHT)
            .unwrap() as i32;
    let width =
        opencv::videoio::VideoCaptureTraitConst::get(&cap, opencv::videoio::CAP_PROP_FRAME_WIDTH)
            .unwrap() as i32;

    // In order to calculate the AMFD, we need to read three frames at a times, the current frame (i_t) is the middle frame, so the first frame in the video will be skipped.
    let mut i_t_minus_1 = opencv::prelude::Mat::default();
    let success = opencv::videoio::VideoCaptureTrait::read(&mut cap, &mut i_t_minus_1).unwrap();

    if !success {
        eprintln!("Unable to read frame from video.");
        std::process::exit(1);
    }

    // The second frame in the video will be the current frame (i_t) and the first to be processed for an output.
    let mut i_t = opencv::prelude::Mat::default();
    let success = opencv::videoio::VideoCaptureTrait::read(&mut cap, &mut i_t).unwrap();

    if !success {
        eprintln!("Unable to read frame from video.");
        std::process::exit(1);
    }

    let mut frame_count = 1;

    // We save the raw frame to disk here because it will be needed in the LRMC process. Yes this is a bit of a hack.
    let frame_path = format!("processing/frames/{}.bmp", frame_count);
    let params = opencv::core::Vector::<i32>::new();
    opencv::imgcodecs::imwrite(&frame_path, &i_t_minus_1, &params).unwrap_or_else(|_| {
        eprintln!("Unable to write frame to file: {}", frame_path);
        std::process::exit(1);
    });

    // To ensure consistency in frame count, we designate the initial black image as the first frame processed by AMFD.
    // Although it may be preferable to use a white frame to prevent LRMC from being disregarded due to bitwise_or, we are uncertain if this will result in a significant impact.
    let black_image = opencv::prelude::Mat::zeros(height, width, opencv::core::CV_8UC3).unwrap();
    let amfd_path = format!("processing/amfd/{}.bmp", frame_count);
    let params = opencv::core::Vector::<i32>::new();
    opencv::imgcodecs::imwrite(&amfd_path, &black_image, &params).unwrap_or_else(|_| {
        eprintln!("Unable to write frame to file: {}", amfd_path);
        std::process::exit(1);
    });

    // The AMFD process is repeated until the end of the video is reached.
    loop {
        frame_count += 1;

        // Get the next frame in the video and set it the next frame (i_t_plus_1).
        let mut i_t_plus_1 = opencv::prelude::Mat::default();
        let success = opencv::videoio::VideoCaptureTrait::read(&mut cap, &mut i_t_plus_1).unwrap();

        // If there are no more frames to read, then we have reached the end of the video and can stop processing.
        if !success {
            // Because we don't have a i_t_plus_1 for the last frame we save a black image to disk to pad the frame count.
            // We also save the raw frame disk (for the hack mentioned previously).
            let frame_path = format!("processing/frames/{}.bmp", frame_count);
            let params = opencv::core::Vector::<i32>::new();
            opencv::imgcodecs::imwrite(&frame_path, &i_t, &params).unwrap_or_else(|_| {
                eprintln!("Unable to write frame to file: {}", frame_path);
                std::process::exit(1);
            });

            let black_image =
                opencv::prelude::Mat::zeros(height, width, opencv::core::CV_8UC3).unwrap();
            let amfd_path = format!("processing/amfd/{}.bmp", frame_count);
            let params = opencv::core::Vector::<i32>::new();
            opencv::imgcodecs::imwrite(&amfd_path, &black_image, &params).unwrap_or_else(|_| {
                eprintln!("Unable to write frame to file: {}", amfd_path);
                std::process::exit(1);
            });

            break;
        }

        // Calculate the differencing images D_{t1}, D_{t2}, and D_{t3}.
        let (dt1, dt2, dt3) = difference_frames(&i_t, &i_t_minus_1, &i_t_plus_1);

        // Calculate the accumulative response image I_d.
        let id = accumulative_response_image(dt1, dt2, dt3);

        // Calculate a threshold T to extract targets.
        let t = calculate_threshold(&id, opt);

        // Convert the accumulative response image to a binary image.
        let mut binary_image = threshold_image(id, t);

        // Perform morphological operations on the binary image.
        morphological_operations(opt, &mut binary_image);

        // Remove false alarms
        remove_false_alarms(&mut binary_image, opt);

        // We must covert the binary image to a color image in order to save it to disk.
        let mut colered_image = opencv::prelude::Mat::default();
        opencv::imgproc::cvt_color(
            &binary_image,
            &mut colered_image,
            opencv::imgproc::COLOR_GRAY2BGR,
            0,
        )
        .unwrap();

        let frame_path = format!("processing/frames/{}.bmp", frame_count);
        let params = opencv::core::Vector::<i32>::new();
        opencv::imgcodecs::imwrite(&frame_path, &i_t, &params).unwrap_or_else(|_| {
            eprintln!("Unable to write frame to file: {}", frame_path);
            std::process::exit(1);
        });

        let amfd_path = format!("processing/amfd/{}.bmp", frame_count);
        let params = opencv::core::Vector::<i32>::new();
        opencv::imgcodecs::imwrite(&amfd_path, &colered_image, &params).unwrap_or_else(|_| {
            eprintln!("Unable to write frame to file: {}", amfd_path);
            std::process::exit(1);
        });

        // Update the frames for the next iteration.
        i_t_minus_1 = i_t;
        i_t = i_t_plus_1;
    }

    opencv::videoio::VideoCaptureTrait::release(&mut cap).unwrap();
}

fn difference_frames(
    i_t: &opencv::prelude::Mat,
    i_t_minus_1: &opencv::prelude::Mat,
    i_t_plus_1: &opencv::prelude::Mat,
) -> (
    opencv::prelude::Mat,
    opencv::prelude::Mat,
    opencv::prelude::Mat,
) {
    // D_{t1} = |I_{t} - I_{t-1}| (Eq. 1)
    let mut dt1 = opencv::prelude::Mat::default();
    opencv::core::absdiff(i_t, i_t_minus_1, &mut dt1).unwrap();

    // D_{t2} = |I_{t+1} - I_{t-1}| (Eq. 2)
    let mut dt2 = opencv::prelude::Mat::default();
    opencv::core::absdiff(i_t_plus_1, i_t_minus_1, &mut dt2).unwrap();

    // D_{t3} = |I_{t+1} - I_{t}| (Eq. 3)
    let mut dt3 = opencv::prelude::Mat::default();
    opencv::core::absdiff(i_t_plus_1, i_t, &mut dt3).unwrap();
    (dt1, dt2, dt3)
}

fn accumulative_response_image(
    dt1: opencv::prelude::Mat,
    dt2: opencv::prelude::Mat,
    dt3: opencv::prelude::Mat,
) -> opencv::prelude::Mat {
    // I_{d} = (D_{t1} + D_{t2} + D_{t3}) / 3 (Eq. 4)
    // Because we are using 8-bit unsigned integers, we need to convert the Mat to 32-bit floating point numbers to prevent overflow.
    // We also convert back to an image but grayscale this time.
    let mut dt1_float = opencv::prelude::Mat::default();
    let mut dt2_float = opencv::prelude::Mat::default();
    let mut dt3_float = opencv::prelude::Mat::default();
    opencv::prelude::MatTraitConst::convert_to(
        &dt1,
        &mut dt1_float,
        opencv::core::CV_32F,
        1.0,
        0.0,
    )
    .unwrap();
    opencv::prelude::MatTraitConst::convert_to(
        &dt2,
        &mut dt2_float,
        opencv::core::CV_32F,
        1.0,
        0.0,
    )
    .unwrap();
    opencv::prelude::MatTraitConst::convert_to(
        &dt3,
        &mut dt3_float,
        opencv::core::CV_32F,
        1.0,
        0.0,
    )
    .unwrap();

    let mut temp1 = opencv::prelude::Mat::default();
    let mut temp2 = opencv::prelude::Mat::default();
    let mut id_float = opencv::prelude::Mat::default();

    opencv::core::add(
        &dt1_float,
        &dt2_float,
        &mut temp1,
        &opencv::core::no_array(),
        -1,
    )
    .unwrap();
    opencv::core::add(
        &temp1,
        &dt3_float,
        &mut temp2,
        &opencv::core::no_array(),
        -1,
    )
    .unwrap();

    opencv::core::multiply(
        &temp2,
        &opencv::core::Scalar::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0),
        &mut id_float,
        1.0,
        opencv::core::CV_32F,
    )
    .unwrap();

    let mut id = opencv::prelude::Mat::default();
    opencv::prelude::MatTraitConst::convert_to(&id_float, &mut id, opencv::core::CV_8U, 1.0, 0.0)
        .unwrap();

    let mut id_gray = opencv::prelude::Mat::default();
    opencv::imgproc::cvt_color(&id, &mut id_gray, opencv::imgproc::COLOR_BGR2GRAY, 0).unwrap();
    id_gray
}

fn calculate_threshold(id: &opencv::prelude::Mat, opt: &crate::Opt) -> f64 {
    // T = mean(I_{d}) + k * std(I_{d}) (Eq. 6)
    let mut mean_val = opencv::prelude::Mat::default();
    let mut std_val = opencv::prelude::Mat::default();
    opencv::core::mean_std_dev(id, &mut mean_val, &mut std_val, &opencv::core::no_array()).unwrap();

    let mean_scalar = opencv::prelude::MatTraitConst::at_2d::<f64>(&mean_val, 0, 0).unwrap();
    let std_scalar = opencv::prelude::MatTraitConst::at_2d::<f64>(&std_val, 0, 0).unwrap();

    let k: f64 = opt.k;
    let t = *mean_scalar + k * *std_scalar;
    t
}

fn threshold_image(id: opencv::prelude::Mat, t: f64) -> opencv::prelude::Mat {
    // I_{d}(x,y) = 255 if I_{d}(x,y) >= T, 0 otherwise (Eq. 5)
    let mut binary_image = opencv::prelude::Mat::default();
    opencv::imgproc::threshold(
        &id,
        &mut binary_image,
        t,
        255.0,
        opencv::imgproc::THRESH_BINARY,
    )
    .unwrap();
    binary_image
}

fn morphological_operations(opt: &crate::Opt, binary_image: &mut opencv::prelude::Mat) {
    let kernel_size_single: i32 = opt.kernel;
    let kernel =
        opencv::prelude::Mat::ones(kernel_size_single, kernel_size_single, opencv::core::CV_8U)
            .unwrap();

    let mut temp_image = binary_image.clone();
    opencv::imgproc::morphology_ex(
        &*binary_image,
        &mut temp_image,
        opencv::imgproc::MORPH_CLOSE,
        &kernel,
        opencv::core::Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        opencv::imgproc::morphology_default_border_value().unwrap(),
    )
    .unwrap();
    *binary_image = temp_image;

    let mut temp_image = binary_image.clone();
    opencv::imgproc::morphology_ex(
        &*binary_image,
        &mut temp_image,
        opencv::imgproc::MORPH_OPEN,
        &kernel,
        opencv::core::Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        opencv::imgproc::morphology_default_border_value().unwrap(),
    )
    .unwrap();
    *binary_image = temp_image;
}

fn remove_false_alarms(binary_image: &mut opencv::prelude::Mat, opt: &crate::Opt) {
    // area_min <= area <= area_max and aspect_ratio_min <= aspect_ratio <= aspect_ratio_max (Eq. 7)
    let mut labels = opencv::prelude::Mat::default();
    let mut stats = opencv::prelude::Mat::default();
    let mut centroids = opencv::prelude::Mat::default();

    let num_labels = opencv::imgproc::connected_components_with_stats(
        &*binary_image,
        &mut labels,
        &mut stats,
        &mut centroids,
        opt.connectivity,
        opencv::core::CV_32S,
    )
    .unwrap();
    for i in 1..num_labels {
        let x = *opencv::prelude::MatTraitConst::at_2d::<i32>(&stats, i as i32, 0).unwrap();
        let y = *opencv::prelude::MatTraitConst::at_2d::<i32>(&stats, i as i32, 1).unwrap();
        let w = *opencv::prelude::MatTraitConst::at_2d::<i32>(&stats, i as i32, 2).unwrap();
        let h = *opencv::prelude::MatTraitConst::at_2d::<i32>(&stats, i as i32, 3).unwrap();
        let area = *opencv::prelude::MatTraitConst::at_2d::<i32>(&stats, i as i32, 4).unwrap();
        let aspect_ratio = w as f64 / h as f64;

        if area < opt.area_min
            || area > opt.area_max
            || aspect_ratio < opt.aspect_ratio_min
            || aspect_ratio > opt.aspect_ratio_max
        {
            for row in y..(y + h) {
                for col in x..(x + w) {
                    if *opencv::prelude::MatTraitConst::at_2d::<i32>(&labels, row, col).unwrap()
                        == i as i32
                    {
                        *opencv::prelude::MatTrait::at_2d_mut::<u8>(binary_image, row, col)
                            .unwrap() = 0;
                    }
                }
            }
        }
    }
}
