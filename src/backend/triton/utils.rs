use super::model_infer_response::InferOutputTensor;

pub(crate) fn get_output_idx(outputs: &[InferOutputTensor], name: &str) -> Option<usize> {
    outputs.iter().position(|v| v.name == name)
}
