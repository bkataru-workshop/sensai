//! AWS SigV4 signing implementation for Bedrock API requests.

use super::BedrockCredentials;
use crate::client::error::LlmError;

use ring::digest;
use ring::hmac;

const SERVICE_NAME: &str = "bedrock";
const AWS_ALGORITHM: &str = "AWS4-HMAC-SHA256";
const AWS4_PREFIX: &str = "AWS4";
const AWS4_REQUEST: &str = "aws4_request";
const CONTENT_TYPE_JSON: &str = "application/json";
const CONTENT_TYPE_EVENT_STREAM: &str = "application/vnd.amazon.eventstream";

/// Sign an HTTP request using AWS SigV4.
///
/// Returns header name/value pairs to include in the request.
pub fn sign_request(
    credentials: &BedrockCredentials,
    region: &str,
    method: &str,
    url: &str,
    body: &str,
    streaming: bool,
) -> Result<Vec<(String, String)>, LlmError> {
    todo!()
}

// == Private Helpers ==

struct ParsedUrl {
    host: String,
    path: String,
    query: Option<String>,
}

fn parse_url(url: &str) -> Result<ParsedUrl, LlmError> {
    let url = url
        .strip_prefix("https://")
        .ok_or_else(|| LlmError::new("INVALID_URL", "URL must start with https://"))?;

    let (host_and_path, query) = match url.split_once('?') {
        Some((hp, q)) => (hp, Some(q.to_string())),
        None => (url, None),
    };

    let (host, path) = match host_and_path.split_once('/') {
        Some((h, p)) => (h.to_string(), format!("/{}", p)),
        None => (host_and_path.to_string(), "/".to_string()),
    };

    Ok(ParsedUrl { host, path, query })
}

fn sha256_hex(data: &[u8]) -> String {
    hex::encode(digest::digest(&digest::SHA256, data).as_ref())
}

fn hmac_key(key: &[u8], data: &[u8]) -> String {
    let key = hmac::Key::new(hmac::HMAC_SHA256, key);
    hex::encode(hmac::sign(&key, data).as_ref())
}

fn hmac_raw(key: &[u8], data: &[u8]) -> Vec<u8> {
    let key = hmac::Key::new(hmac::HMAC_SHA256, key);
    hmac::sign(&key, data).as_ref().to_vec()
}

fn derive_signing_key(secret: &str, data: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_raw(
        format!("{}{}", AWS4_PREFIX, secret).as_bytes(),
        data.as_bytes(),
    );
    let k_region = hmac_raw(&k_date, region.as_bytes());
    let k_service = hmac_raw(&k_region, service.as_bytes());
    hmac_raw(&k_service, AWS4_REQUEST.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_url() {
        let p = parse_url("https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse")
            .unwrap();
        assert_eq!(p.host, "bedrock-runtime.us-east-1.amazonaws.com");
        assert_eq!(p.path, "/model/test/converse");
        assert!(p.query.is_none());
    }

    #[test]
    fn test_parse_url_with_query() {
        let p = parse_url("https://example.com/path?foo=bar").unwrap();
        assert_eq!(p.host, "example.com");
        assert_eq!(p.path, "/path");
        assert_eq!(p.query.as_deref(), Some("foo=bar"));
    }

    #[test]
    fn test_sha256_empty() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sign_request_produces_required_headers() {
        let creds = BedrockCredentials::new("AKID_TEST", "SECRET_TEST");
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse";

        let headers = sign_request(&creds, "us-east-1", "POST", url, "{}", false).unwrap();
        let names: Vec<&str> = headers.iter().map(|(k, _)| k.as_str()).collect();

        assert!(names.contains(&"Authorization"));
        assert!(names.contains(&"X-Amz-Date"));
        assert!(names.contains(&"X-Amz-Content-Sha256"));
        assert!(names.contains(&"Host"));
        assert!(names.contains(&"Content-Type"));
        assert!(names.contains(&"Accept"));
        // No security token header when session_token is None
        assert!(!names.contains(&"X-Amz-Security-Token"));
    }

    #[test]
    fn test_sign_request_with_session_token() {
        let creds = BedrockCredentials::with_session_token("AKID", "SECRET", "TOKEN");
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse";

        let headers = sign_request(&creds, "us-east-1", "POST", url, "{}", false).unwrap();
        let names: Vec<&str> = headers.iter().map(|(k, _)| k.as_str()).collect();

        assert!(names.contains(&"X-Amz-Security-Token"));
    }

    #[test]
    fn test_sign_request_streaming_accept() {
        let creds = BedrockCredentials::new("AKID", "SECRET");
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse-stream";

        let headers = sign_request(&creds, "us-east-1", "POST", url, "{}", true).unwrap();
        let accept = headers.iter().find(|(k, _)| k == "Accept").unwrap();

        assert_eq!(accept.1, "application/vnd.amazon.eventstream");
    }
}
