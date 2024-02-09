use std::fs::File;
use std::io::Read;

use criterion::{criterion_group, criterion_main, Criterion};
use serde::Deserialize;

use ai_router::backend::triton::routes::embeddings::{
    build_embedding_response_data, transform_triton_f32_array,
};

#[derive(Deserialize)]
struct TestTransformTritonF32ArrayData {
    input_batch_size_1: Vec<u8>,
    input_batch_size_4: Vec<u8>,
    output_batch_size_1: Vec<Vec<f32>>,
    output_batch_size_4: Vec<Vec<f32>>,
}

fn bench_triton_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_triton");

    let mut test_data = String::new();

    File::open("tests/backend.triton.routes.embeddings.test_transform_triton_f32_array")
        .unwrap()
        .read_to_string(&mut test_data)
        .unwrap();

    let test_data: TestTransformTritonF32ArrayData = serde_json::from_str(&test_data).unwrap();

    group.bench_function("transform_triton_f32_array_batch_size_1", |b| {
        b.iter(|| transform_triton_f32_array(&test_data.input_batch_size_1, 1, 1024));
    });

    group.bench_function("transform_triton_f32_array_batch_size_4", |b| {
        b.iter(|| transform_triton_f32_array(&test_data.input_batch_size_4, 4, 1024));
    });

    group.bench_function("build_embedding_response_data.batch_size_1", |b| {
        b.iter(|| build_embedding_response_data(&test_data.output_batch_size_1));
    });

    group.bench_function("build_embedding_response_data.batch_size_4", |b| {
        b.iter(|| build_embedding_response_data(&test_data.output_batch_size_4));
    });
}

criterion_group!(benches, bench_triton_embeddings);
criterion_main!(benches);
