use std::{net::SocketAddr, time::Duration};

use axum::body::{Body, HttpBody};
use axum::http::Request;
use axum::{extract::ConnectInfo, response::Response};
use tracing::field::display;
use tracing::{field::Empty, Span};

pub fn tl_access_log_make_span(request: &Request<Body>) -> Span {
    tracing::info_span!(
        "request",
        // http_version = display(request.version()),
        method = %request.method(),
        request_time = Empty,
        remote = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ci| display(ci.ip())),
        size = Empty,
        status = Empty,
        ts = Empty,
        user = Empty,
        uri = %request.uri(),
    )
}

pub fn tl_on_reponse(response: &Response, request_time: Duration, span: &Span) {
    span.record("request_time", display(format!("{request_time:?}")));
    span.record(
        "size",
        display(response.body().size_hint().exact().unwrap_or(0)),
    );
    span.record("status", display(response.status().as_u16()));
}
