/**
 * Text-to-speech helper using the Web Speech API.
 * Note: availability and voice quality depend on the device OS/browser.
 * For kiosk deployments, voice packs can be provisioned at OS level (not as offline bundles in this UI).
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
