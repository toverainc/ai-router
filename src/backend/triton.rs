#![allow(clippy::pedantic)]
tonic::include_proto!("inference");

pub(crate) mod request;
pub mod routes;
pub(crate) mod utils;
