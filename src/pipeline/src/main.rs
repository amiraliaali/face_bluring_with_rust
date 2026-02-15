mod helpers;

use anyhow::Result;
use helpers::{black_out, draw_rectangle, nms, Detection, mat_to_rgba, rgba_to_mat};
use image::io::Reader as ImageReader;
use image::{GenericImageView, Rgba, RgbaImage};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{Array4, ArrayD, Axis};
use onnxruntime::GraphOptimizationLevel;
use opencv::{
    core,
    prelude::*,
    videoio,
};

fn main() -> Result<()> {
    let environment = Environment::builder().with_name("face_blur").build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_number_threads(4)?
        .with_model_from_file("yolov8n-face.onnx")?;

    println!("Model loaded successfully!");

    let mut cap = videoio::VideoCapture::from_file(
        "test_videos/vid_3.mp4",
        videoio::CAP_ANY,
    )?;

    if !cap.is_opened()? {
        panic!("Cannot open video");
    }

    let mut frame = core::Mat::default();
    cap.read(&mut frame)?;
    if frame.empty() {
        panic!("Video is empty");
    }

    let frame_size = frame.size()?;

    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;

    let mut writer = videoio::VideoWriter::new(
        "test_videos/vid_3_processed.mp4",
        fourcc,
        30.0,
        frame.size()?,
        true,
    )?;

    loop {
        cap.read(&mut frame)?;

        if frame.empty(){
            break;
        }

        // println!("Processed frame!");

        let mut output_img = mat_to_rgba(&frame)?;
        let (orig_width, orig_height) = (output_img.width(), output_img.height());
        let resized_img = image::imageops::resize(
            &output_img,
            640,
            640,
            image::imageops::FilterType::Triangle,
        );


        let img_tensor: Array4<f32> = Array4::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| {
            let pixel = resized_img.get_pixel(x as u32, y as u32);
            pixel[c] as f32 / 255.0
        });

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
            black_out(
                &mut output_img,
                det.x1 as u32,
                det.y1 as u32,
                det.x2 as u32,
                det.y2 as u32,
            );
        }

        let frame_to_write = rgba_to_mat(&output_img)?;
        writer.write(&frame_to_write)?;

    }

    Ok(())
}
