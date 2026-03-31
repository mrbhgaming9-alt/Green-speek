/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef } from 'react';
import { GoogleGenAI, Modality } from "@google/genai";
import { Volume2, Play, Square, Loader2, Github, Youtube, MessageSquare, Download } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export default function App() {
  const [text, setText] = useState('');
  const [targetLanguage, setTargetLanguage] = useState('Urdu');
  const [selectedVoice, setSelectedVoice] = useState('Kore');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const pcmDataRef = useRef<Int16Array | null>(null);

  const stopAudio = () => {
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop();
      } catch (e) {
        // Ignore if already stopped
      }
      sourceNodeRef.current = null;
    }
    setIsPlaying(false);
  };

  const playPCM = async (data: Int16Array) => {
    stopAudio();

    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    }

    const ctx = audioContextRef.current;
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const buffer = ctx.createBuffer(1, data.length, 24000);
    const channelData = buffer.getChannelData(0);
    
    for (let i = 0; i < data.length; i++) {
      channelData[i] = data[i] / 32768.0;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => setIsPlaying(false);
    
    sourceNodeRef.current = source;
    source.start();
    setIsPlaying(true);
  };

  const downloadAudio = () => {
    if (!pcmDataRef.current) return;
    
    const data = pcmDataRef.current;
    const buffer = new ArrayBuffer(44 + data.length * 2);
    const view = new DataView(buffer);
    
    // RIFF identifier
    view.setUint32(0, 0x52494646, false); // "RIFF"
    // file length
    view.setUint32(4, 36 + data.length * 2, true);
    // RIFF type
    view.setUint32(8, 0x57415645, false); // "WAVE"
    // format chunk identifier
    view.setUint32(12, 0x666d7420, false); // "fmt "
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (raw)
    view.setUint16(20, 1, true);
    // channel count
    view.setUint16(22, 1, true);
    // sample rate
    view.setUint32(24, 24000, true);
    // byte rate (sample rate * block align)
    view.setUint32(28, 24000 * 2, true);
    // block align (channel count * bytes per sample)
    view.setUint16(32, 2, true);
    // bits per sample
    view.setUint16(34, 16, true);
    // data chunk identifier
    view.setUint32(36, 0x64617461, false); // "data"
    // data chunk length
    view.setUint32(40, data.length * 2, true);
    
    // write the PCM samples
    for (let i = 0; i < data.length; i++) {
      view.setInt16(44 + i * 2, data[i], true);
    }
    
    const blob = new Blob([buffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `greenspeak_${selectedVoice}_${Date.now()}.wav`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleTranslate = async () => {
    if (!text.trim()) return;
    setIsTranslating(true);
    setError(null);

    try {
      const apiKey = process.env.GEMINI_API_KEY;
      const ai = new GoogleGenAI({ apiKey: apiKey as string });
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [{ parts: [{ text: `Translate the following text to ${targetLanguage}. Only provide the translated text, nothing else: "${text}"` }] }],
      });

      const translatedText = response.text;
      if (translatedText) {
        setText(translatedText.trim());
      } else {
        throw new Error('Translation failed. No text returned.');
      }
    } catch (err: any) {
      console.error('Translation Error:', err);
      setError(`Translation Error: ${err.message || 'Failed to translate.'}`);
    } finally {
      setIsTranslating(false);
    }
  };

  const handleGenerateTTS = async () => {
    if (!text.trim()) return;

    setIsGenerating(true);
    setError(null);
    stopAudio();
    pcmDataRef.current = null;

    try {
      const apiKey = process.env.GEMINI_API_KEY;
      const ai = new GoogleGenAI({ apiKey: apiKey as string });
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: text }] }],
        config: {
          responseModalities: ["AUDIO" as any],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: selectedVoice },
            },
          },
        },
      });

      const candidate = response.candidates?.[0];
      if (!candidate) {
        throw new Error('The model blocked the response or returned no candidates. This usually happens due to safety filters.');
      }

      if (candidate.finishReason && !['STOP', 'MAX_TOKENS'].includes(candidate.finishReason)) {
        throw new Error(`Generation failed. Reason: ${candidate.finishReason}`);
      }

      // Search through all parts for audio data
      let base64Audio = '';
      const parts = candidate.content?.parts || [];
      for (const part of parts) {
        if (part.inlineData?.data) {
          base64Audio = part.inlineData.data;
          break;
        }
      }
      
      if (base64Audio) {
        const binary = atob(base64Audio);
        const buffer = new ArrayBuffer(binary.length);
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const pcmData = new Int16Array(buffer);
        pcmDataRef.current = pcmData;
        await playPCM(pcmData);
      } else {
        // Check if there's text instead (maybe a safety message or refusal)
        let textResponse = '';
        try {
          textResponse = response.text || '';
        } catch (e) {
          // response.text might throw if there's no text part
        }
        
        if (textResponse) {
          throw new Error(`Model returned text instead of audio data: ${textResponse}`);
        }
        throw new Error('The model response did not contain any audio data. Please try different text or a different voice.');
      }
    } catch (err: any) {
      console.error('TTS Generation Error (v1.2):', err);
      // Extract more specific error message if available
      const errorMessage = err.message || 'Failed to generate speech. Please try again.';
      setError(`Error (v1.2): ${errorMessage}`);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 sm:p-8">
      {/* Top Branding */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed top-0 left-0 right-0 p-4 flex justify-center items-center bg-emerald-50/80 backdrop-blur-md z-10 border-b border-emerald-100"
      >
        <a 
          href="https://www.youtube.com/@mrbhgaming" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-emerald-700 font-bold tracking-tight hover:text-emerald-500 transition-colors group"
        >
          <Youtube className="w-5 h-5 group-hover:scale-110 transition-transform" />
          <span>@mrbhgaming</span>
        </a>
      </motion.div>

      <main className="w-full max-w-2xl mt-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="bg-white rounded-3xl shadow-2xl shadow-emerald-200/50 overflow-hidden border border-emerald-100"
        >
          <div className="p-8 sm:p-12">
            <header className="mb-8 text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-emerald-100 rounded-2xl mb-4">
                <Volume2 className="w-8 h-8 text-emerald-600" />
              </div>
              <h1 className="text-3xl font-bold text-emerald-900 mb-2">GreenSpeak</h1>
              <a 
                href="https://www.youtube.com/@mrbhgaming" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-emerald-600 font-medium hover:text-emerald-500 transition-colors"
              >
                Text-to-Speech by Mr bh gaming
              </a>
            </header>

              <div className="space-y-6">
                <div className="flex flex-col gap-4 p-4 bg-emerald-50/30 rounded-2xl border border-emerald-100">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-bold text-emerald-800">Translate to:</label>
                    <select 
                      value={targetLanguage}
                      onChange={(e) => setTargetLanguage(e.target.value)}
                      className="bg-white border border-emerald-200 rounded-lg px-3 py-1 text-sm text-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      {['Urdu', 'Hindi', 'English', 'Arabic', 'French', 'Spanish', 'German', 'Chinese', 'Japanese', 'Russian', 'Bengali', 'Punjabi'].map(lang => (
                        <option key={lang} value={lang}>{lang}</option>
                      ))}
                    </select>
                  </div>
                  <button
                    onClick={handleTranslate}
                    disabled={isTranslating || !text.trim()}
                    className="w-full py-2 bg-emerald-100 hover:bg-emerald-200 disabled:bg-emerald-50 text-emerald-700 rounded-xl font-bold text-sm transition-all flex items-center justify-center gap-2"
                  >
                    {isTranslating ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Translating...</span>
                      </>
                    ) : (
                      <>
                        <MessageSquare className="w-4 h-4" />
                        <span>Translate Text</span>
                      </>
                    )}
                  </button>
                </div>

                <div className="flex flex-wrap gap-2 justify-center">
                {['Puck', 'Charon', 'Kore', 'Fenrir', 'Zephyr'].map((voice) => (
                  <button
                    key={voice}
                    onClick={() => setSelectedVoice(voice)}
                    className={`px-4 py-2 rounded-xl text-sm font-bold transition-all ${
                      selectedVoice === voice
                        ? 'bg-emerald-600 text-white shadow-md'
                        : 'bg-emerald-50 text-emerald-600 hover:bg-emerald-100'
                    }`}
                  >
                    {voice}
                  </button>
                ))}
              </div>

              <div className="relative">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Type something smooth here..."
                  className="w-full h-40 p-6 bg-emerald-50/50 border-2 border-emerald-100 rounded-2xl focus:border-emerald-500 focus:ring-0 transition-all resize-none text-lg text-emerald-900 placeholder:text-emerald-300"
                />
                <div className="absolute bottom-4 right-4 text-emerald-400 text-sm font-mono">
                  {text.length} characters
                </div>
              </div>

              <AnimatePresence mode="wait">
                {error && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="p-4 bg-red-50 text-red-600 rounded-xl text-sm font-medium border border-red-100"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={handleGenerateTTS}
                  disabled={isGenerating || !text.trim()}
                  className="flex-1 h-16 bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-200 text-white rounded-2xl font-bold text-lg transition-all flex items-center justify-center gap-3 shadow-lg shadow-emerald-600/20 active:scale-95"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      <span>Generating...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-6 h-6 fill-current" />
                      <span>Convert to Speech</span>
                    </>
                  )}
                </button>

                {pcmDataRef.current && (
                  <div className="flex gap-4">
                    <motion.button
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      onClick={() => isPlaying ? stopAudio() : playPCM(pcmDataRef.current!)}
                      className="flex-1 h-16 bg-emerald-100 hover:bg-emerald-200 text-emerald-700 rounded-2xl font-bold transition-all flex items-center justify-center gap-3 active:scale-95"
                    >
                      {isPlaying ? (
                        <>
                          <Square className="w-6 h-6 fill-current" />
                          <span>Stop</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-6 h-6 fill-current" />
                          <span>Replay</span>
                        </>
                      )}
                    </motion.button>

                    <motion.button
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      onClick={downloadAudio}
                      className="h-16 px-8 bg-emerald-50 hover:bg-emerald-100 text-emerald-600 rounded-2xl font-bold transition-all flex items-center justify-center gap-3 active:scale-95 border border-emerald-100"
                      title="Download Audio"
                    >
                      <Download className="w-6 h-6" />
                      <span className="hidden sm:inline">Download</span>
                    </motion.button>
                  </div>
                )}
              </div>

              {isPlaying && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2 pt-4"
                >
                  <div className="flex-1 h-1 bg-emerald-100 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: "0%" }}
                      animate={{ width: "100%" }}
                      transition={{ duration: (pcmDataRef.current?.length || 0) / 24000, ease: "linear" }}
                      className="h-full bg-emerald-500"
                    />
                  </div>
                  <span className="text-xs font-mono text-emerald-500">Playing...</span>
                </motion.div>
              )}
            </div>
          </div>

          <footer className="bg-emerald-50 p-6 flex justify-between items-center border-t border-emerald-100">
            <div className="text-emerald-600 text-xs font-medium flex flex-col">
              <span>Powered by Gemini 2.5 Flash</span>
              <span className="opacity-50">App Version 1.3</span>
            </div>
            <div className="flex gap-4">
              <a href="#" className="text-emerald-400 hover:text-emerald-600 transition-colors">
                <MessageSquare className="w-5 h-5" />
              </a>
              <a href="#" className="text-emerald-400 hover:text-emerald-600 transition-colors">
                <Github className="w-5 h-5" />
              </a>
            </div>
          </footer>
        </motion.div>
      </main>

      <footer className="mt-12 text-emerald-400 text-sm font-medium">
        © 2026 <a href="https://www.youtube.com/@mrbhgaming" target="_blank" rel="noopener noreferrer" className="hover:text-emerald-600 transition-colors underline underline-offset-4">Mr bh gaming</a>. All rights reserved.
      </footer>
    </div>
  );
}
