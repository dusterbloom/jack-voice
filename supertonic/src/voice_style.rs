// Supertonic TTS - Voice Style Management
// Handles loading and managing speaker embeddings (style_ttl and style_dp)

use ndarray::Array3;
use serde::Deserialize;
use std::path::Path;

/// Voice style information
#[derive(Clone, Debug, serde::Serialize)]
pub struct VoiceStyle {
    /// Voice identifier (e.g., "F1", "M1")
    pub id: String,
    /// Human-readable name
    pub name: String,
}

/// JSON structure for voice style files from HuggingFace
/// Format: {"style_ttl": {"data": [[[f32, ...]]]}, "style_dp": {"data": [[[f32, ...]]]}}
#[derive(Debug, Deserialize)]
struct VoiceStyleJson {
    style_ttl: StyleData,
    style_dp: StyleData,
}

#[derive(Debug, Deserialize)]
struct StyleData {
    data: Vec<Vec<Vec<f32>>>, // 3D array: [batch][rows][cols]
}

/// Voice style data with both style embeddings
/// - style_ttl: Used by text encoder and vector estimator [1, 50, 256]
/// - style_dp: Used by duration predictor [1, 8, 16]
#[derive(Clone, Debug)]
pub struct VoiceStyleData {
    /// Voice style info
    pub style: VoiceStyle,
    /// Text-to-latent style embeddings (shape: [1, 50, 256])
    pub style_ttl: Array3<f32>,
    /// Duration predictor style embeddings (shape: [1, 8, 16])
    pub style_dp: Array3<f32>,
}

impl VoiceStyleData {
    /// Load voice style from a JSON file (HuggingFace format)
    pub fn from_json_file<P: AsRef<Path>>(
        path: P,
        id: &str,
        name: &str,
    ) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let json: VoiceStyleJson = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Parse style_ttl: [batch][rows][cols] -> Array3<f32>
        let style_ttl = parse_3d_array(&json.style_ttl.data)?;

        // Parse style_dp: [batch][rows][cols] -> Array3<f32>
        let style_dp = parse_3d_array(&json.style_dp.data)?;

        log::info!(
            "Loaded voice style {}: style_ttl {:?}, style_dp {:?}",
            id,
            style_ttl.shape(),
            style_dp.shape()
        );

        Ok(Self {
            style: VoiceStyle {
                id: id.to_string(),
                name: name.to_string(),
            },
            style_ttl,
            style_dp,
        })
    }

    /// Get style_ttl as a flat Vec<f32> for ONNX input
    pub fn style_ttl_flat(&self) -> Vec<f32> {
        self.style_ttl.iter().copied().collect()
    }

    /// Get style_dp as a flat Vec<f32> for ONNX input
    pub fn style_dp_flat(&self) -> Vec<f32> {
        self.style_dp.iter().copied().collect()
    }

    /// Get style_ttl shape as [batch, rows, cols]
    pub fn style_ttl_shape(&self) -> [usize; 3] {
        let shape = self.style_ttl.shape();
        [shape[0], shape[1], shape[2]]
    }

    /// Get style_dp shape as [batch, rows, cols]
    pub fn style_dp_shape(&self) -> [usize; 3] {
        let shape = self.style_dp.shape();
        [shape[0], shape[1], shape[2]]
    }
}

/// Parse a 3D Vec into an ndarray Array3
fn parse_3d_array(data: &[Vec<Vec<f32>>]) -> Result<Array3<f32>, std::io::Error> {
    if data.is_empty() || data[0].is_empty() || data[0][0].is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Empty style data",
        ));
    }

    let dim0 = data.len();
    let dim1 = data[0].len();
    let dim2 = data[0][0].len();

    // Flatten the data in row-major order
    let flat: Vec<f32> = data.iter().flatten().flatten().copied().collect();

    Array3::from_shape_vec((dim0, dim1, dim2), flat).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to create array: {}", e),
        )
    })
}

/// Voice style registry for managing multiple voices
#[allow(dead_code)]
pub struct VoiceRegistry {
    styles: Vec<VoiceStyleData>,
    default_style: Option<String>,
}

#[allow(dead_code)]
impl VoiceRegistry {
    /// Create a new voice registry
    pub fn new() -> Self {
        Self {
            styles: Vec::new(),
            default_style: None,
        }
    }

    /// Add a voice style to the registry
    pub fn add(&mut self, style_data: VoiceStyleData) {
        if self.default_style.is_none() {
            self.default_style = Some(style_data.style.id.clone());
        }
        self.styles.push(style_data);
    }

    /// Get a voice style by ID
    pub fn get(&self, id: &str) -> Option<&VoiceStyleData> {
        self.styles.iter().find(|s| s.style.id == id)
    }

    /// Get the default voice style
    pub fn get_default(&self) -> Option<&VoiceStyleData> {
        self.default_style.as_ref().and_then(|id| self.get(id))
    }

    /// Set the default voice style
    pub fn set_default(&mut self, id: &str) {
        if self.get(id).is_some() {
            self.default_style = Some(id.to_string());
        }
    }

    /// List all available voices
    pub fn list(&self) -> Vec<&VoiceStyle> {
        self.styles.iter().map(|s| &s.style).collect()
    }

    /// Load voices from a directory (JSON format only)
    pub fn load_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<usize, std::io::Error> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Ok(0);
        }

        let mut count = 0;

        // Look for JSON voice files in the voices directory
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |ext| ext == "json") {
                if let Some(stem) = path.file_stem() {
                    let id = stem.to_string_lossy().to_string();
                    let name = format_voice_name(&id);

                    match VoiceStyleData::from_json_file(&path, &id, &name) {
                        Ok(style) => {
                            log::info!(
                                "Loaded voice: {} ({}) - ttl: {:?}, dp: {:?}",
                                id,
                                name,
                                style.style_ttl.shape(),
                                style.style_dp.shape()
                            );
                            self.add(style);
                            count += 1;
                        }
                        Err(e) => {
                            log::warn!("Failed to load voice {}: {}", id, e);
                        }
                    }
                }
            }
        }

        Ok(count)
    }
}

impl Default for VoiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Format voice ID as human-readable name
#[allow(dead_code)]
fn format_voice_name(id: &str) -> String {
    match id {
        "F1" => "Female Voice 1".to_string(),
        "F2" => "Female Voice 2".to_string(),
        "M1" => "Male Voice 1".to_string(),
        "M2" => "Male Voice 2".to_string(),
        _ => {
            // Convert ID to title case
            let mut result = String::new();
            for (i, ch) in id.chars().enumerate() {
                if i == 0 || id.chars().nth(i - 1) == Some('_') {
                    result.extend(ch.to_uppercase());
                } else if ch == '_' {
                    result.push(' ');
                } else {
                    result.push(ch);
                }
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_voice_name() {
        assert_eq!(format_voice_name("F1"), "Female Voice 1");
        assert_eq!(format_voice_name("custom_voice"), "Custom Voice");
    }

    #[test]
    fn test_voice_registry() {
        let registry = VoiceRegistry::new();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_parse_3d_array() {
        let data = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
        let arr = parse_3d_array(&data).unwrap();
        assert_eq!(arr.shape(), &[1, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 1, 1]], 4.0);
    }
}
