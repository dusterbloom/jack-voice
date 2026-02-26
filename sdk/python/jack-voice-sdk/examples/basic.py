from jack_voice_sdk import JackVoice


def main() -> None:
    client = JackVoice.connect()
    try:
        frame_16k_pcm = b"\x00\x00" * 320
        utterance_16k_pcm = b"\x00\x00" * 3200

        print(client.vad(frame_16k_pcm))
        print(client.stt(utterance_16k_pcm, language="auto"))
        print(client.tts("Build finished.", engine="kokoro", voice="35"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
