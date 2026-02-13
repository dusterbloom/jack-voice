// Voice pipeline watchdog helpers (timeouts for long-running operations)

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutEvent {
    pub elapsed: Duration,
    pub limit: Duration,
}

/// Simple timeout tracker that fires once per start() call.
#[derive(Debug)]
pub struct TimeoutTracker {
    limit: Duration,
    started_at: Option<Instant>,
    fired: bool,
}

impl TimeoutTracker {
    pub fn new(limit: Duration) -> Self {
        Self {
            limit,
            started_at: None,
            fired: false,
        }
    }

    pub fn start(&mut self, now: Instant) {
        self.started_at = Some(now);
        self.fired = false;
    }

    pub fn stop(&mut self) {
        self.started_at = None;
        self.fired = false;
    }

    pub fn check(&mut self, now: Instant) -> Option<TimeoutEvent> {
        let started_at = self.started_at?;
        if self.fired {
            return None;
        }
        let elapsed = now.duration_since(started_at);
        if elapsed > self.limit {
            self.fired = true;
            return Some(TimeoutEvent {
                elapsed,
                limit: self.limit,
            });
        }
        None
    }

    pub fn is_running(&self) -> bool {
        self.started_at.is_some()
    }
}

pub fn speaking_timeout_exceeded(started_at: Instant, now: Instant, limit: Duration) -> bool {
    now.duration_since(started_at) > limit
}

/// Determine whether dead-mic detection should be suppressed.
///
/// Dead-mic should NOT accumulate when:
/// - TTS is actively playing (`is_speaking_state`)
/// - We're in the BT grace period after TTS ends (`in_grace_period`)
/// - User is mid-speech / SmartTurn is evaluating (`in_speech`) — BT earbuds
///   duty-cycle zero frames between packets during speech
pub fn should_suppress_dead_mic(
    is_speaking_state: bool,
    in_grace_period: bool,
    in_speech: bool,
) -> bool {
    is_speaking_state || in_grace_period || in_speech
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_tracker_fires_once_after_limit() {
        let mut tracker = TimeoutTracker::new(Duration::from_secs(2));
        let start = Instant::now();
        tracker.start(start);

        assert!(tracker.check(start + Duration::from_secs(1)).is_none());

        let fired = tracker.check(start + Duration::from_secs(3));
        assert!(fired.is_some());

        // Subsequent checks should not fire again until restarted
        let fired_again = tracker.check(start + Duration::from_secs(4));
        assert!(fired_again.is_none());
    }

    #[test]
    fn timeout_tracker_resets_on_stop() {
        let mut tracker = TimeoutTracker::new(Duration::from_secs(1));
        let start = Instant::now();
        tracker.start(start);
        tracker.stop();

        let fired = tracker.check(start + Duration::from_secs(2));
        assert!(fired.is_none());
        assert!(!tracker.is_running());
    }

    #[test]
    fn speaking_timeout_helper() {
        let now = Instant::now();
        let started = now - Duration::from_secs(5);
        assert!(speaking_timeout_exceeded(
            started,
            now,
            Duration::from_secs(3)
        ));
        assert!(!speaking_timeout_exceeded(
            started,
            now,
            Duration::from_secs(10)
        ));
    }

    /// Simulates the playback-tail watchdog refresh pattern:
    /// After all TTS chunks are synthesized and queued to rodio, the inner loop
    /// refreshes started_at each iteration. This prevents the 45s watchdog from
    /// firing while audio is still playing.
    #[test]
    fn speaking_timeout_resets_with_refresh() {
        let limit = Duration::from_secs(45);

        // Simulate: synthesis finishes at T=0, audio plays for 60s.
        // Without refresh, watchdog fires at T=45.
        // With periodic refresh every ~10ms, it never fires.
        let t0 = Instant::now();

        // At T=44s (no refresh yet), timeout has NOT fired
        let started_at = t0;
        let check_time = t0 + Duration::from_secs(44);
        assert!(!speaking_timeout_exceeded(started_at, check_time, limit));

        // At T=46s without refresh, timeout WOULD fire
        let check_time = t0 + Duration::from_secs(46);
        assert!(speaking_timeout_exceeded(started_at, check_time, limit));

        // But with a refresh at T=44s, the new started_at is T=44s
        let refreshed_at = t0 + Duration::from_secs(44);
        // At T=46s, only 2s since refresh — no timeout
        assert!(!speaking_timeout_exceeded(refreshed_at, check_time, limit));
        // At T=90s (46s since refresh), timeout fires
        let check_time = t0 + Duration::from_secs(90);
        assert!(speaking_timeout_exceeded(refreshed_at, check_time, limit));
    }

    /// Dead mic detection must be suppressed during active speech (in_speech=true).
    /// BT earbuds send zero frames between packets during SmartTurn evaluation,
    /// which would falsely trigger dead-mic restart without this guard.
    #[test]
    fn dead_mic_suppressed_during_speech() {
        // Not speaking, no grace, no speech → dead mic should accumulate
        assert!(!should_suppress_dead_mic(false, false, false));

        // TTS playing → suppress
        assert!(should_suppress_dead_mic(true, false, false));

        // Grace period after TTS → suppress
        assert!(should_suppress_dead_mic(false, true, false));

        // User mid-speech (SmartTurn evaluating) → suppress
        assert!(should_suppress_dead_mic(false, false, true));

        // All true → suppress
        assert!(should_suppress_dead_mic(true, true, true));
    }
}
