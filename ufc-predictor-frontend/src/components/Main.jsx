import React, { useState, useEffect, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

// Parses "Fighter Name will likely win (73.2% confidence)" into parts.
// Falls back gracefully to just showing the raw message if the format
// ever changes on the backend.
function parsePrediction(message) {
  const match = message.match(
    /^(.+?)\s+will likely win\s+\(([\d.]+)%\s+confidence\)$/i,
  );
  if (!match) return { winner: null, confidence: null, raw: message };
  return { winner: match[1], confidence: parseFloat(match[2]), raw: message };
}

// Animates a number counting up from 0 to `target` whenever `target` changes.
function useCountUp(target, duration = 900) {
  const [value, setValue] = useState(0);
  const frameRef = useRef(null);

  useEffect(() => {
    if (target === null || target === undefined) {
      setValue(0);
      return;
    }
    const start = performance.now();
    const from = 0;

    const tick = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(from + (target - from) * eased);
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(tick);
      } else {
        setValue(target);
      }
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [target, duration]);

  return value;
}

const Main = () => {
  const [redFighter, setRedFighter] = useState("");
  const [blueFighter, setBlueFighter] = useState("");
  const [fighters, setFighters] = useState([]);
  const [redSuggestions, setRedSuggestions] = useState([]);
  const [blueSuggestions, setBlueSuggestions] = useState([]);
  const [prediction, setPrediction] = useState(null); // { winner, confidence, raw }
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("Error occurred");

  const animatedConfidence = useCountUp(prediction?.confidence ?? null);

  useEffect(() => {
    fetch(`${API_URL}/api/fighters`)
      .then((res) => res.json())
      .then((data) => setFighters(data.fighters || []))
      .catch(() => {});
  }, []);

  const handleRedChange = (e) => {
    const val = e.target.value;
    setRedFighter(val);
    if (val.length > 1) {
      setRedSuggestions(
        fighters.filter((f) => f.includes(val.toLowerCase())).slice(0, 5),
      );
    } else {
      setRedSuggestions([]);
    }
  };

  const handleBlueChange = (e) => {
    const val = e.target.value;
    setBlueFighter(val);
    if (val.length > 1) {
      setBlueSuggestions(
        fighters.filter((f) => f.includes(val.toLowerCase())).slice(0, 5),
      );
    } else {
      setBlueSuggestions([]);
    }
  };

  const handleSwap = () => {
    setRedFighter(blueFighter);
    setBlueFighter(redFighter);
    setRedSuggestions([]);
    setBlueSuggestions([]);
  };

  const handlePredict = async () => {
    if (!redFighter.trim() || !blueFighter.trim()) {
      setIsError(true);
      setErrorMessage("Both fighter names are required.");
      return;
    }

    setIsError(false);
    setIsLoading(true);
    setPrediction(null);

    const payload = { red: redFighter.trim(), blue: blueFighter.trim() };
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeout);
      const data = await res.json();

      if (!res.ok || data.error) {
        setIsError(true);
        if (data.error && data.error.includes("Fighter Not Found")) {
          setErrorMessage(
            "One or both fighters not found. Check the spelling.",
          );
        } else if (data.error && data.error.includes("two different")) {
          setErrorMessage("Enter two different fighters.");
        } else {
          setErrorMessage("Network error. Try again.");
        }
        setIsLoading(false);
        return;
      }

      if (!data.message || data.message.toLowerCase().includes("not found")) {
        setIsError(true);
        setErrorMessage("One or both fighters not found. Check the spelling.");
        return;
      }

      setPrediction(parsePrediction(data.message));
      setIsError(false);
    } catch (e) {
      clearTimeout(timeout);
      setIsError(true);
      if (e.name === "AbortError") {
        setErrorMessage("Server inactive. Try again in 60 seconds.");
      } else {
        setErrorMessage(
          "Something went wrong. Check your inputs and try again.",
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-start md:items-center justify-center px-4 py-10 md:py-16">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8 bg-black/60 rounded-xl p-5">
          <p className="text-emerald-400 text-xs font-bold tracking-widest uppercase mb-2">
            UFC Fight predictor
          </p>
          <h1 className="text-white text-2xl md:text-3xl font-bold mb-3">
            Who wins this matchup?
          </h1>
          <p className="text-gray-200 text-sm leading-relaxed max-w-md mx-auto">
            Enter both fighters to get the model's pick, based on career stats,
            recent form, and matchup history.
          </p>
        </div>
        {/* Card */}
        <div className="bg-[#161616] border border-white/10 rounded-xl">
          {/* Corners */}
          <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr]">
            <div className="p-5 border-b md:border-b-0 md:border-r border-white/10">
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="w-2 h-2 rounded-full bg-red-500"
                  aria-hidden="true"
                />
                <p className="text-gray-100 text-sm font-semibold">
                  Red corner
                </p>
              </div>
              <div className="relative">
                <input
                  className="bg-[#0d0d0d] text-white placeholder-gray-500 text-sm rounded-md px-3 py-2.5 w-full border border-white/10 focus:outline-none focus:border-red-500/60 transition-colors"
                  type="text"
                  placeholder="Fighter name"
                  value={redFighter}
                  onChange={handleRedChange}
                  autoComplete="off"
                />
                {redSuggestions.length > 0 && (
                  <ul className="absolute z-50 top-full mt-1 bg-[#1c1c1c] border border-white/10 w-full rounded-md shadow-lg max-h-48 overflow-y-auto animate-fade-down">
                    {redSuggestions.map((name) => (
                      <li
                        key={name}
                        className="px-3 py-2 hover:bg-white/5 cursor-pointer text-gray-100 text-sm capitalize border-b border-white/5 last:border-0"
                        onClick={() => {
                          setRedFighter(name);
                          setRedSuggestions([]);
                        }}
                      >
                        {name}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* VS divider — the circular badge is the ONLY bordered element here,
                no partial container border competing with it */}
            <div className="flex md:flex-col items-center justify-center gap-1.5 py-3 md:py-0 md:px-5">
              <div className="border border-white/15 rounded-full w-9 h-9 flex items-center justify-center bg-[#0d0d0d] shrink-0">
                <span className="text-gray-300 text-xs font-bold">VS</span>
              </div>
              <button
                onClick={handleSwap}
                className="text-emerald-400 hover:text-emerald-300 text-xs font-medium transition-colors cursor-pointer"
              >
                swap
              </button>
            </div>

            <div className="p-5 border-t md:border-t-0 md:border-l border-white/10">
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="w-2 h-2 rounded-full bg-blue-500"
                  aria-hidden="true"
                />
                <p className="text-gray-100 text-sm font-semibold">
                  Blue corner
                </p>
              </div>
              <div className="relative">
                <input
                  className="bg-[#0d0d0d] text-white placeholder-gray-500 text-sm rounded-md px-3 py-2.5 w-full border border-white/10 focus:outline-none focus:border-blue-500/60 transition-colors"
                  type="text"
                  placeholder="Fighter name"
                  value={blueFighter}
                  onChange={handleBlueChange}
                  autoComplete="off"
                />
                {blueSuggestions.length > 0 && (
                  <ul className="absolute z-50 top-full mt-1 bg-[#1c1c1c] border border-white/10 w-full rounded-md shadow-lg max-h-48 overflow-y-auto animate-fade-down">
                    {blueSuggestions.map((name) => (
                      <li
                        key={name}
                        className="px-3 py-2 hover:bg-white/5 cursor-pointer text-gray-100 text-sm capitalize border-b border-white/5 last:border-0"
                        onClick={() => {
                          setBlueFighter(name);
                          setBlueSuggestions([]);
                        }}
                      >
                        {name}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          {/* Error */}
          {isError && (
            <div className="border-t border-white/10 bg-red-500/10 px-5 py-3">
              <p className="text-red-300 text-sm text-center font-medium">
                {errorMessage}
              </p>
            </div>
          )}

          {/* Predict */}
          <div className="border-t border-white/10 p-5">
            <button
              onClick={handlePredict}
              disabled={isLoading}
              className="bg-emerald-500 hover:bg-emerald-400 disabled:bg-emerald-800 disabled:cursor-not-allowed text-black text-sm font-bold rounded-md py-2.5 w-full transition-colors flex items-center justify-center gap-2 cursor-pointer"
            >
              {isLoading && (
                <span
                  className="h-3.5 w-3.5 rounded-full border-2 border-black/30 border-t-black animate-spin"
                  aria-hidden="true"
                />
              )}
              {isLoading ? "Predicting" : "Predict winner"}
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Predicted winner */}
          <div className="bg-[#161616] border border-white/10 rounded-xl p-5 text-center">
            <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide mb-1.5">
              Predicted winner
            </p>

            <p
              className={`min-h-[48px] flex items-center justify-center ${
                prediction && !isLoading
                  ? "text-white font-bold animate-fade-down text-3xl m-4"
                  : "text-gray-500 text-sm m-4"
              }`}
            >
              {isLoading
                ? "Crunching the numbers..."
                : prediction
                  ? (prediction.winner ?? prediction.raw)
                  : "Enter two fighters above to see a prediction"}
            </p>
          </div>

          {/* Model confidence */}
          <div className="bg-[#161616] border border-white/10 rounded-xl p-5 text-center">
            <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide mb-3">
              Model confidence
            </p>

            <p className="text-emerald-400 text-4xl font-bold text-center mb-3 tabular-nums min-h-[48px] flex items-center justify-center">
              {prediction?.confidence != null && !isLoading
                ? `${animatedConfidence.toFixed(1)}%`
                : "—"}
            </p>

            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full bg-emerald-500 transition-[width] duration-100 ease-out"
                style={{
                  width:
                    prediction?.confidence != null && !isLoading
                      ? `${animatedConfidence}%`
                      : "0%",
                }}
              />
            </div>
          </div>
        </div>

        <p className="text-gray-300 text-xs text-center mt-6 bg-black/50 rounded-lg py-2 px-3">
          Model accuracy is roughly 65&ndash;73% depending on fighter
          experience. Server sleeps after 15 minutes of inactivity, so the first
          request may take a moment.
        </p>
      </div>
    </div>
  );
};

export default Main;
