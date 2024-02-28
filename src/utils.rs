use std::fmt;
use std::marker::PhantomData;
use std::str;
use std::str::Utf8Error;

use bytes::{Buf, Bytes};
use serde::{de, Deserialize, Deserializer};

pub(crate) fn string_or_seq_string<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVec(PhantomData<Vec<String>>);

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_owned()])
        }

        fn visit_seq<S>(self, visitor: S) -> Result<Self::Value, S::Error>
        where
            S: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(visitor))
        }
    }

    deserializer.deserialize_any(StringOrVec(PhantomData))
}

pub(crate) fn deserialize_bytes_tensor(encoded_tensor: Vec<u8>) -> Result<Vec<String>, Utf8Error> {
    let mut bytes = Bytes::from(encoded_tensor);
    let mut strs = Vec::new();
    while bytes.has_remaining() {
        let len = bytes.get_u32_le() as usize;
        if len <= bytes.remaining() {
            let slice = bytes.split_to(len);
            let s = str::from_utf8(&slice)?;
            strs.push(s.to_string());
        }
    }
    Ok(strs)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;

    use serde::Deserialize;

    use super::deserialize_bytes_tensor;

    #[derive(Deserialize)]
    struct UtilsTestData {
        input: Vec<u8>,
        output: Vec<String>,
    }

    #[test]
    fn test_deserialize_bytes_tensor() {
        const TESTDATA_FILE: &str = "tests/utils.deserialize_bytes_tensor";

        let mut test_data = String::new();

        File::open(TESTDATA_FILE)
            .unwrap_or_else(|e| panic!("failed to open testdata file '{TESTDATA_FILE}': {e}"))
            .read_to_string(&mut test_data)
            .unwrap_or_else(|e| panic!("failed to read testdata file '{TESTDATA_FILE}': {e}"));
        let test_data: UtilsTestData =
            serde_json::from_str(&test_data).expect("failed to convert testdata to JSON");

        let test_result =
            deserialize_bytes_tensor(test_data.input).expect("failed to deserialize testdata");

        assert_eq!(test_result, test_data.output);
    }
}
