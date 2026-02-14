use jack_voice::{TtsEngine, VoicePipelineConfig};

#[test]
fn voice_pipeline_defaults_to_pocket() {
    let config = VoicePipelineConfig::default();
    assert_eq!(config.tts_engine, TtsEngine::Pocket);
    assert_eq!(config.tts_voice, "alba");
}
