mod helpers;

use anyhow::Result;
use helpers::{nms, preload_ort_global, Detection};

use ndarray::{Array4, ArrayD, ArrayViewD, Axis, IxDyn};

use ort::value::Value;
use ort::{ep, session::Session};

use opencv::imgproc::FILLED;
use opencv::{core, dnn, prelude::*, videoio};

use ort::sys;

fn main() -> Result<()> {
    preload_ort_global();

    let mut session = Session::builder()?
        .with_execution_providers([ep::CUDA::default().build()])?
        .commit_from_file("yolov8n-face.onnx")?;

    println!("Model loaded successfully!");

    let input_name = session.inputs()[0].name().to_string();

    let mut cap = videoio::VideoCapture::from_file("test_videos/vid_3.mp4", videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Cannot open video");
    }

    let mut frame = core::Mat::default();
    cap.read(&mut frame)?;
    if frame.empty() {
        panic!("Video is empty");
    }

    let orig_width = frame.cols();
    let orig_height = frame.rows();

    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = videoio::VideoWriter::new(
        "test_videos/vid_3_processed.mp4",
        fourcc,
        30.0,
        frame.size()?,
        true,
    )?;

    let mut blob = core::Mat::default();

    loop {
        cap.read(&mut frame)?;
        if frame.empty() {
            break;
        }

        blob = dnn::blob_from_image(
            &frame,
            1.0 / 255.0,
            core::Size::new(640, 640),
            core::Scalar::default(), // mean
            true,                    // swapRB: BGR->RGB
            false,                   // crop
            core::CV_32F,
        )?;

        let blob_data: &[f32] = blob.data_typed()?;


        let dims = blob.mat_size();
        let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();

        let input_view = ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&shape), blob_data)
            .expect("blob shape mismatch");


        let input_value = ort::value::Value::from_array(input_view.to_owned())?;

        let outputs = session.run(vec![(input_name.as_str(), input_value)])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let predictions: ArrayViewD<'_, f32> =
            ArrayViewD::from_shape(IxDyn(&dims), data).expect("Invalid output shape");

        let preds = predictions.index_axis(Axis(0), 0);
        let preds = preds.t();

        let mut detections: Vec<Detection> = Vec::new();
        for row in preds.axis_iter(Axis(0)) {
            let cx = row[0];
            let cy = row[1];
            let w = row[2];
            let h = row[3];
            let conf = row[4];

            if conf < 0.4 {
                continue;
            }

            let x1 = ((cx - w / 2.0) / 640.0 * orig_width as f32).max(0.0);
            let y1 = ((cy - h / 2.0) / 640.0 * orig_height as f32).max(0.0);
            let x2 = ((cx + w / 2.0) / 640.0 * orig_width as f32).min(orig_width as f32 - 1.0);
            let y2 = ((cy + h / 2.0) / 640.0 * orig_height as f32).min(orig_height as f32 - 1.0);

            detections.push(Detection {
                x1,
                y1,
                x2,
                y2,
                conf,
            });
        }

        let final_detections = nms(detections, 0.45);

        for det in final_detections {
            let roi = core::Rect::new(
                det.x1 as i32,
                det.y1 as i32,
                (det.x2 - det.x1) as i32,
                (det.y2 - det.y1) as i32,
            );

            opencv::imgproc::rectangle(
                &mut frame,
                roi,
                core::Scalar::all(0.0),
                FILLED,
                opencv::imgproc::LINE_8,
                0,
            )?;
        }

        writer.write(&frame)?;
    }

    Ok(())
}
