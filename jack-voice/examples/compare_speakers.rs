use jack_voice::{models, TextToSpeech, TtsEngine};
use std::time::Instant;

fn main() {
    // Ensure magpie model is ready
    if !models::magpie_model_ready() {
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        runtime
            .block_on(models::ensure_magpie_model(&models::NoopProgress))
            .expect("ensure magpie model");
    }

    let test_text = "Ciao a tutti, questa è una prova.";
    println!("\n=== Comparing Speaker 0 vs 2 ===");
    println!("Text: {}\n", test_text);

    // Test speaker 0 (current default)
    println!("--- Speaker 0 (current default) ---");
    let mut tts0 = TextToSpeech::with_engine(TtsEngine::Magpie).expect("init");
    tts0.set_language("it").expect("set lang");
    tts0.set_speaker("0").expect("set speaker");

    let start = Instant::now();
    let audio0 = tts0.synthesize(test_text).expect("synth");
    let time0 = start.elapsed();

    let dur0 = audio0.samples.len() as f32 / audio0.sample_rate as f32;
    println!("  Time: {:?}", time0);
    println!("  Duration: {:.2}s", dur0);
    println!("  RTF: {:.2}x", time0.as_secs_f32() / dur0);

    // Test speaker 2 (fastest from previous test)
    println!("\n--- Speaker 2 (fastest candidate) ---");
    let mut tts2 = TextToSpeech::with_engine(TtsEngine::Magpie).expect("init");
    tts2.set_language("it").expect("set lang");
    tts2.set_speaker("2").expect("set speaker");

    let start = Instant::now();
    let audio2 = tts2.synthesize(test_text).expect("synth");
    let time2 = start.elapsed();

    let dur2 = audio2.samples.len() as f32 / audio2.sample_rate as f32;
    println!("  Time: {:?}", time2);
    println!("  Duration: {:.2}s", dur2);
    println!("  RTF: {:.2}x", time2.as_secs_f32() / dur2);

    println!("\nRecommendation:");
    if time2 < time0 {
        let speedup = time0.as_secs_f32() / time2.as_secs_f32();
        println!("  Speaker 2 is {:.1}x faster than Speaker 0", speedup);
        println!("  Consider changing default from '0' to '2'");
    } else {
        println!("  Speaker 0 is fine as default");
    }
}
