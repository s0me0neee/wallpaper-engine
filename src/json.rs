//! Package JSON as Wallpaper Engine reads it, which is not quite JSON.
//!
//! WE's reader accepts a trailing comma, and its own stock `fluidsimulation/effect.json` ends its
//! `dependencies` array with one; strict parsing dropped the whole effect chain of every layer that used
//! it (2967697841's sky). Of 3985 packaged JSON files in an 87-scene library that is the only failure.

use serde::Deserialize;
use serde::de::DeserializeOwned;

pub fn from_slice<T: DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    let mut deserializer = serde_json_lenient::Deserializer::from_slice(bytes);
    deserializer.set_ignore_trailing_commas(true);
    // Through a `Value`: the lenient parser forgets the option while skipping a field the target lacks.
    let value = serde_json::Value::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(serde_json::from_value(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trailing_comma_parses() {
        let value: serde_json::Value = from_slice(b"{\"dependencies\": [\"a\", \"b\",\r\n\t], \"passes\": [1, ], }").unwrap();
        assert_eq!(value["dependencies"][1], "b");
        assert_eq!(value["passes"][0], 1);
    }

    #[test]
    fn a_trailing_comma_in_a_skipped_field_parses() {
        #[derive(serde::Deserialize)]
        struct Passes {
            passes: Vec<u32>,
        }
        let value: Passes = from_slice(b"{\"dependencies\": [\"a\",\r\n\t], \"passes\": [1]}").unwrap();
        assert_eq!(value.passes, [1]);
    }
}
