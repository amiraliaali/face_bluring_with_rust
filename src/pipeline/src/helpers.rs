use image::{Rgba, RgbaImage};
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::{MatTraitConst, MatTraitConstManual, MatTrait};
use anyhow::Result;



#[derive(Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub conf: f32,
}

pub fn draw_rectangle(img: &mut RgbaImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgba<u8>) {
    let (width, height) = img.dimensions();

    for x in x1..=x2.min(width - 1) {
        if y1 < height {
            img.put_pixel(x, y1, color);
        }
        if y2 < height {
            img.put_pixel(x, y2, color);
        }
    }

    for y in y1..=y2.min(height - 1) {
        if x1 < width {
            img.put_pixel(x1, y, color);
        }
        if x2 < width {
            img.put_pixel(x2, y, color);
        }
    }
}

pub fn black_out(img: &mut RgbaImage, x1: u32, y1: u32, x2: u32, y2: u32) {
    let (width, height) = img.dimensions();
    let color: Rgba<u8> = Rgba([0, 0, 0, 255]);

    for x in x1..=x2.min(width - 1) {
        for y in y1..=y2.min(height - 1) {
            img.put_pixel(x, y, color);
        }
    }
}

pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    inter_area / (area_a + area_b - inter_area + 1e-6)
}

pub fn nms(mut dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap());

    let mut result = Vec::new();

    while let Some(best) = dets.pop() {
        dets.retain(|d| iou(&best, d) < iou_thresh);
        result.push(best);
    }

    result
}


pub fn mat_to_rgba(mat: &Mat) -> Result<RgbaImage> {
    let mut mat_rgba = Mat::default();
    imgproc::cvt_color(mat, &mut mat_rgba, imgproc::COLOR_BGR2RGBA, 0)?;

    let (width, height) = (mat_rgba.cols(), mat_rgba.rows());
    let data = mat_rgba.data_bytes()?;

    let img = RgbaImage::from_raw(width as u32, height as u32, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("Failed to convert Mat to RgbaImage"))?;

    Ok(img)
}

pub fn rgba_to_mat(img: &RgbaImage) -> Result<Mat> {
    let img_vec = img.to_vec();

    let mat = Mat::from_slice(&img_vec)?;

    let mat = mat.reshape(4, img.height() as i32)?;

    let mut mat_bgr = Mat::default();
    imgproc::cvt_color(&mat, &mut mat_bgr, imgproc::COLOR_RGBA2BGR, 0)?;

    Ok(mat_bgr)
}
