type Point = (i32, i32);
type BBox = (i32, i32, i32, i32);
type Object = (Point, BBox);

pub fn pf(opt: &crate::Opt) {
    let video_file = &opt.video_file;
    let mut cap = opencv::videoio::VideoCapture::from_file(video_file, 0).unwrap();

    if !opencv::videoio::VideoCaptureTraitConst::is_opened(&cap).unwrap() {
        eprintln!("Unable to open video: {}", video_file);
        std::process::exit(1);
    }

    let frame_count =
        opencv::videoio::VideoCaptureTraitConst::get(&cap, opencv::videoio::CAP_PROP_FRAME_COUNT)
            .unwrap() as i32;

    opencv::videoio::VideoCaptureTrait::release(&mut cap).unwrap();

    let file = std::fs::File::create("labels.csv").unwrap();
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_writer(std::io::BufWriter::new(file));
    wtr.write_record(&["frame", "x1", "y1", "x2", "y2"])
        .unwrap();
    wtr.flush().unwrap();

    if std::path::Path::new("output").exists() {
        std::fs::remove_dir_all("output").unwrap();
    }
    std::fs::create_dir_all("output").unwrap();

    let mut objects: Vec<Option<Vec<Object>>> =
        vec![None; (opt.pipeline_length + 1).try_into().unwrap()];

    for abs_current_image_idx in 0..(frame_count as usize - opt.pipeline_length as usize) {
        get_frame_objs(&mut objects, abs_current_image_idx);
        
        if objects[0].as_ref().map_or(0, |o| o.len()) > 0{
            let mut current_frame_obj_to_next_frame_obj_lookup: Vec<Vec<(usize, usize)>> =
                Vec::new();
            let mut next_frames_obj_correspondences =
                vec![
                    vec![0; opt.pipeline_length as usize + 1];
                    objects[0].as_ref().map_or(0, |o| o.len())
                ];

            (1..=opt.pipeline_length as usize).for_each(|rel_next_frame_idx| {
                let mut cost_matrix: Vec<Vec<i32>> =
                    vec![
                        vec![0; objects[rel_next_frame_idx].as_ref().map_or(0, |o| o.len())];
                        objects[0].as_ref().map_or(0, |o| o.len())
                    ];
                if let Some(current_objects) = &objects[0] {
                    for (a, (point_a, _)) in current_objects.iter().enumerate() {
                        if let Some(next_frame_objects) = &objects[rel_next_frame_idx] {
                            for (b, (point_b, _)) in next_frame_objects.iter().enumerate() {
                                let sab_x = (point_a.0 - point_b.0).abs();
                                let sab_y = (point_a.1 - point_b.1).abs();
                                if 0 < sab_x
                                    && sab_x < opt.pipeline_size
                                    && 0 < sab_y
                                    && sab_y < opt.pipeline_size
                                {
                                    cost_matrix[a][b] = 1;
                                }
                            }
                        }
                    }
                }

                loop {
                    if !(cost_matrix.len() > cost_matrix[0].len()) {
                        break;
                    }
                    let mut new_cost_matrix = vec![vec![0; cost_matrix.len()]; cost_matrix.len()];
                    for (i, row) in cost_matrix.iter().enumerate() {
                        for (j, col) in row.iter().enumerate() {
                            new_cost_matrix[i][j] = *col;
                        }
                    }
                    cost_matrix = new_cost_matrix;
                }

                let cash_flow_as_matrx =
                    pathfinding::matrix::Matrix::from_rows(cost_matrix).unwrap();
                let (_, assignment) = pathfinding::prelude::kuhn_munkres(&cash_flow_as_matrx);
                let mut optimal_correspondances: Vec<(usize, usize)> = Vec::new();

                for a_idx in 0..cash_flow_as_matrx.rows {
                    let b_idx = assignment[a_idx];
                    if cash_flow_as_matrx[(a_idx, b_idx)] == 1 {
                        optimal_correspondances.push((a_idx, b_idx));
                    }
                }

                current_frame_obj_to_next_frame_obj_lookup.push(optimal_correspondances.clone());
                for (a_idx, _) in optimal_correspondances {
                    next_frames_obj_correspondences[a_idx][rel_next_frame_idx - 1] = 1;
                }
            });

            (0..objects[0].as_ref().map_or(0, |o| o.len())).for_each(
                |current_frame_current_center_idx| {
                    if next_frames_obj_correspondences[current_frame_current_center_idx]
                        .iter()
                        .sum::<usize>()
                        > opt.h.try_into().unwrap()
                    {
                        let mut pointline: Vec<Option<Object>> =
                            vec![None; (opt.pipeline_length + 1).try_into().unwrap()];

                        pointline[0] = objects[0]
                            .as_ref()
                            .map(|o| o[current_frame_current_center_idx]);

                        for (rel_next_frame_idx, results) in
                            current_frame_obj_to_next_frame_obj_lookup
                                .iter()
                                .enumerate()
                        {
                            let next_center_idx = results
                                .iter()
                                .find(|(a, _)| *a == current_frame_current_center_idx)
                                .map(|(_, b)| *b);

                            if let Some(next_center_idx) = next_center_idx {
                                pointline[rel_next_frame_idx + 1] = objects[rel_next_frame_idx + 1]
                                    .as_ref()
                                    .map(|o| o[next_center_idx]);
                            }
                        }

                        for rel_all_frame_idx in 1..opt.pipeline_length {
                            if pointline[rel_all_frame_idx as usize].is_none() {
                                let next_non_none_index = pointline
                                    .iter()
                                    .enumerate()
                                    .skip(rel_all_frame_idx as usize + 1)
                                    .find(|(_, v)| v.is_some())
                                    .map(|(i, _)| i);

                                let prev_non_none_index = pointline
                                    .iter()
                                    .enumerate()
                                    .rev()
                                    .skip(opt.pipeline_length as usize - rel_all_frame_idx as usize)
                                    .find(|(_, v)| v.is_some())
                                    .map(|(i, _)| i);

                                if next_non_none_index.is_none() {
                                    if let Some(prev_non_none_index) = prev_non_none_index {
                                        if prev_non_none_index > 0 {
                                            let x_prev =
                                                pointline[prev_non_none_index - 1].unwrap().0 .0;
                                            let y_prev =
                                                pointline[prev_non_none_index - 1].unwrap().0 .1;
                                            let x_cur =
                                                pointline[prev_non_none_index].unwrap().0 .0;
                                            let y_cur =
                                                pointline[prev_non_none_index].unwrap().0 .1;
                                            let delta_x = x_cur - x_prev;
                                            let delta_y = y_cur - y_prev;
                                            let extrapolated_center =
                                                (x_cur + delta_x, y_cur + delta_y);

                                            let bbox_prev =
                                                pointline[prev_non_none_index - 1].unwrap().1;
                                            let bbox_cur =
                                                pointline[prev_non_none_index].unwrap().1;
                                            let delta_bbox = (
                                                bbox_cur.0 - bbox_prev.0,
                                                bbox_cur.1 - bbox_prev.1,
                                                bbox_cur.2 - bbox_prev.2,
                                                bbox_cur.3 - bbox_prev.3,
                                            );
                                            let extrapolated_bbox = (
                                                bbox_cur.0 + delta_bbox.0,
                                                bbox_cur.1 + delta_bbox.1,
                                                bbox_cur.2 + delta_bbox.2,
                                                bbox_cur.3 + delta_bbox.3,
                                            );

                                            pointline[rel_all_frame_idx as usize] =
                                                Some((extrapolated_center, extrapolated_bbox));
                                            if let Some(objects_vec) =
                                                &mut objects[rel_all_frame_idx as usize]
                                            {
                                                objects_vec
                                                    .push((extrapolated_center, extrapolated_bbox));
                                            }
                                        }
                                    }
                                } else {
                                    if let Some(next_non_none_index) = next_non_none_index {
                                        if let Some(prev_non_none_index) = prev_non_none_index {
                                            let (x0, y0) =
                                                pointline[prev_non_none_index].unwrap().0;
                                            let (x1, y1) =
                                                pointline[next_non_none_index].unwrap().0;

                                            let delta_x = (x1 - x0)
                                                / (next_non_none_index - prev_non_none_index)
                                                    as i32;
                                            let delta_y = (y1 - y0)
                                                / (next_non_none_index - prev_non_none_index)
                                                    as i32;

                                            let extrapolated_center = (
                                                x0 + delta_x
                                                    * (rel_all_frame_idx as i32
                                                        - prev_non_none_index as i32),
                                                y0 + delta_y
                                                    * (rel_all_frame_idx as i32
                                                        - prev_non_none_index as i32),
                                            );

                                            let bbox0 = pointline[prev_non_none_index].unwrap().1;
                                            let bbox1 = pointline[next_non_none_index].unwrap().1;

                                            let average_bbox = (
                                                (bbox0.0 + bbox1.0) / 2,
                                                (bbox0.1 + bbox1.1) / 2,
                                                (bbox0.2 + bbox1.2) / 2,
                                                (bbox0.3 + bbox1.3) / 2,
                                            );

                                            pointline[rel_all_frame_idx as usize] =
                                                Some((extrapolated_center, average_bbox));

                                            if let Some(objects_vec) =
                                                &mut objects[rel_all_frame_idx as usize]
                                            {
                                                objects_vec
                                                    .push((extrapolated_center, average_bbox));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if let Some(ref mut objects_vec) = &mut objects[0] {
                            if let Some(object) =
                                objects_vec.get_mut(current_frame_current_center_idx)
                            {
                                object.0 = (-1, -1);
                                object.1 = (-1, -1, -1, -1);
                            }
                        }
                    }
                },
            );
        }

        let mut color_image = opencv::imgcodecs::imread(
            format!("processing/frames/{}.bmp", abs_current_image_idx + 1).as_str(),
            opencv::imgcodecs::IMREAD_COLOR,
        )
        .unwrap();

        if let Some(objects_vec) = &objects[0] {
            for (point, bbox) in objects_vec.iter() {
                if point.0 != -1 && bbox.0 != -1 {
                    opencv::imgproc::rectangle(
                        &mut color_image,
                        opencv::core::Rect::new(bbox.0, bbox.1, bbox.2, bbox.3),
                        opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                        2,
                        opencv::imgproc::LINE_8,
                        0,
                    )
                    .unwrap();

                    opencv::imgproc::circle(
                        &mut color_image,
                        opencv::core::Point::new(point.0, point.1),
                        2,
                        opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        2,
                        opencv::imgproc::LINE_8,
                        0,
                    )
                    .unwrap();

                    wtr.serialize((
                        abs_current_image_idx,
                        bbox.0,
                        bbox.1,
                        bbox.0 + bbox.2,
                        bbox.1 + bbox.3,
                    ))
                    .unwrap();

                    wtr.flush().unwrap();
                }
            }
        }

        opencv::imgcodecs::imwrite(
            format!("output/{}.bmp", abs_current_image_idx + 1).as_str(),
            &color_image,
            &opencv::core::Vector::default(),
        )
        .unwrap();

        for i in 0..opt.pipeline_length as usize {
            objects[i] = objects[i + 1].clone();
        }

        objects[opt.pipeline_length as usize] = None;
    }
}

fn get_frame_objs(
    objects: &mut Vec<Option<Vec<((i32, i32), (i32, i32, i32, i32))>>>,
    abs_current_image_idx: usize,
) {
    objects
        .iter_mut()
        .enumerate()
        .for_each(|(frame_object_idx, frame_objects)| {
            if frame_objects.is_none() {
                let (centers_frame, bboxes_frame) = find_centers_and_bboxes(
                    (abs_current_image_idx + frame_object_idx + 1)
                        .try_into()
                        .unwrap(),
                )
                .unwrap_or_else(|_| (Vec::new(), Vec::new()));

                let objects_frame: Vec<Object> = centers_frame
                    .iter()
                    .zip(bboxes_frame.iter())
                    .map(|(center, rect)| {
                        (
                            (center.x, center.y),
                            (rect.x, rect.y, rect.width, rect.height),
                        )
                    })
                    .collect();

                *frame_objects = Some(objects_frame);
            }
        });
}

pub fn find_centers_and_bboxes(
    image_num: i32,
) -> opencv::Result<(Vec<opencv::core::Point>, Vec<opencv::core::Rect>)> {
    let amfd_image = opencv::imgcodecs::imread(
        format!("processing/amfd/{}.bmp", image_num).as_str(),
        opencv::imgcodecs::IMREAD_GRAYSCALE,
    )?;

    let lrmc_image = opencv::imgcodecs::imread(
        format!("processing/lrmc/{}.bmp", image_num).as_str(),
        opencv::imgcodecs::IMREAD_GRAYSCALE,
    )?;

    let mut merged_image = opencv::prelude::Mat::default();
    opencv::core::bitwise_or(
        &amfd_image,
        &lrmc_image,
        &mut merged_image,
        &opencv::core::no_array(),
    )?;

    let mut contours = opencv::types::VectorOfVectorOfPoint::new();

    opencv::imgproc::find_contours(
        &merged_image,
        &mut contours,
        opencv::imgproc::RETR_EXTERNAL,
        opencv::imgproc::CHAIN_APPROX_SIMPLE,
        opencv::core::Point::default(),
    )?;

    let mut centers = Vec::new();
    let mut bboxes = Vec::new();

    for contour in contours.iter() {
        let mut contour_mat = opencv::prelude::Mat::new_rows_cols_with_default(
            1,
            contour.len() as i32,
            opencv::core::CV_32SC2,
            opencv::core::Scalar::default(),
        )?;

        {
            let contour_mat_data = opencv::prelude::MatTraitManual::data_typed_mut::<
                opencv::core::Point,
            >(&mut contour_mat)?;
            for (i, point) in contour.iter().enumerate() {
                contour_mat_data[i] = point;
            }
        }

        let m = opencv::imgproc::moments(&contour_mat, false)?;
        let center = if m.m00 != 0.0 {
            opencv::core::Point::new((m.m10 / m.m00) as i32, (m.m01 / m.m00) as i32)
        } else {
            opencv::core::Point::default()
        };
        centers.push(center);

        let rect = opencv::imgproc::bounding_rect(&contour_mat)?;
        bboxes.push(rect);
    }

    // delete images from disk
    std::fs::remove_file(format!("processing/amfd/{}.bmp", image_num)).unwrap();
    std::fs::remove_file(format!("processing/lrmc/{}.bmp", image_num)).unwrap();

    Ok((centers, bboxes))
}
