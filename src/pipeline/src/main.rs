use onnxruntime::environment::Environment;
use onnxruntime::GraphOptimizationLevel;
use anyhow::Result;

fn main() -> Result<()> {
    let environment = Environment::builder()
        .with_name("face_blur")
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_number_threads(4)?
        .with_model_from_file("yolov8n-face.onnx")?;

    println!("Model loaded successfully!");

    Ok(())
}
