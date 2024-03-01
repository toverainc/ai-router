use super::model_infer_response::InferOutputTensor;

pub fn get_output_idx(outputs: &[InferOutputTensor], name: &str) -> Option<usize> {
    outputs.iter().position(|v| v.name == name)
}
