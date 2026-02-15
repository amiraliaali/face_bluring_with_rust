mod helpers;

use anyhow::Result;
use helpers::{nms, Detection};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{Array4, ArrayD, Axis};
use onnxruntime::GraphOptimizationLevel;
use opencv::imgproc::FILLED;

use opencv::{core, prelude::*, videoio};

fn main() -> Result<()> {
    let environment = Environment::builder().with_name("face_blur").build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_number_threads(4)?
        .with_model_from_file("yolov8n-face.onnx")?;

    println!("Model loaded successfully!");

    let mut cap = videoio::VideoCapture::from_file("test_videos/vid_4.mp4", videoio::CAP_ANY)?;

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
        "test_videos/vid_4_processed.mp4",
        fourcc,
        30.0,
        frame.size()?,
        true,
    )?;

    loop {
        cap.read(&mut frame)?;

        if frame.empty() {
            break;
        }

        // println!("Processed frame!");

        let mut resized_frame = Mat::default();
        opencv::imgproc::resize(
            &frame,
            &mut resized_frame,
            core::Size {
                width: 640,
                height: 640,
            },
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        let mut img_tensor: Array4<f32> = Array4::zeros((1, 3, 640, 640));

        for y in 0..640 {
            for x in 0..640 {
                let pixel = resized_frame.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                img_tensor[[0, 0, y as usize, x as usize]] = pixel[2] as f32 / 255.0; // R
                img_tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0; // G
                img_tensor[[0, 2, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                // B
            }
        }

        let input_tensor: ArrayD<f32> = img_tensor.into_dyn();
        // println!("Image prepared for inference.");

        let outputs = session.run::<f32, f32, _>(vec![input_tensor])?;
        let predictions = &outputs[0].view();
        // println!("Predictions shape: {:?}", predictions.shape());

        let preds = predictions.index_axis(Axis(0), 0);
        let preds = preds.t();

        let mut detections: Vec<Detection> = Vec::new();

        for row in preds.axis_iter(Axis(0)) {
            let cx = row[0];
            let cy = row[1];
            let w = row[2];
            let h = row[3];

            let conf = row[4];

            if conf < 0.5 {
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

        // println!("Detections before NMS: {}", detections.len());

        let final_detections = nms(detections, 0.45);

        // println!("Detections after NMS: {}", final_detections.len());

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
