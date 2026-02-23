use image::{Rgba, RgbaImage};
use std::ffi::{CStr, CString};

#[derive(Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub conf: f32,
}

pub fn preload_ort_global() {

    let ort_core = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| {
        "/home/amirali/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2".to_string()
    });
    // export ORT_DYLIB_PATH=/home/amirali/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2
    // export LD_LIBRARY_PATH=/home/amirali/.local/lib/python3.10/site-packages/onnxruntime/capi:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    let ort_dir = std::path::Path::new(&ort_core)
        .parent()
        .unwrap()
        .to_string_lossy()
        .to_string();

    let providers_shared = format!("{}/libonnxruntime_providers_shared.so", ort_dir);

    unsafe {
        libc::dlerror();

        for p in [&ort_core, &providers_shared] {
            let c_path = CString::new(p.as_str()).unwrap();
            let handle = libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);

            if handle.is_null() {
                let err = libc::dlerror();
                let msg = if err.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                };
                eprintln!("Preload failed: {}\n  dlerror: {}", p, msg);
            } else {
                println!("Preloaded (GLOBAL): {}", p);
            }
        }
    }
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
