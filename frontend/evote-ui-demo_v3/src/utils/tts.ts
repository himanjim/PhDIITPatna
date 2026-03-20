/**
 * Text-to-speech helper for optional accessibility support.
 *
 * The helper delegates speech synthesis to the browser and operating system.
 * Availability, voice quality, and installed language coverage therefore depend
 * on the execution environment rather than on bundled application assets.
 */

/**
 * Speak the supplied text using the browser speech-synthesis interface when that
 * interface is available.
 */
export function speak(text: string, lang: string = "en-IN") {
  try {
    const synth = window.speechSynthesis;
    if (!synth) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang;
    synth.cancel();
    synth.speak(u);
  } catch {
    // Fail silently; TTS is an optional enhancement.
  }
}
