use image::{Rgba, RgbaImage};

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
