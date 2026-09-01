import React, { useState, useEffect, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

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

// One row in a factor list: a label + a bar sized by relative importance.
// Bar width is scaled against the list's own max so the strongest factor
// in whichever list (long or short) always reads as "full".
function FactorRow({ factor, maxImportance, color }) {
  const widthPct =
    maxImportance > 0 ? (factor.importance / maxImportance) * 100 : 0;

  return (
    <li className="flex items-center gap-3">
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between gap-2 mb-1">
          <span className="text-gray-100 text-sm capitalize truncate">
            {factor.label}
          </span>

          <span
            className={`text-${color}-400 text-[10px] font-semibold uppercase tracking-wide shrink-0`}
          >
            {factor.strength}
          </span>
        </div>

        <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
          <div
            className={`h-full bg-${color}-500 rounded-full transition-[width] duration-500 ease-out`}
            style={{ width: `${widthPct}%` }}
          />
        </div>
      </div>
    </li>
  );
}

// Renders a full dynamic-length factor list (no fixed top-N cap).
// If the list is long, only the first COLLAPSE_AT show by default with
// a "show all" toggle — this keeps a 2-factor fight compact and an
// 8-factor fight fully visible without either looking broken.
const COLLAPSE_AT = 4;

function FactorList({ title, emptyText, factors, color, maxImportance }) {
  const [expanded, setExpanded] = useState(false);

  if (!factors || factors.length === 0) {
    return (
      <div className="p-5">
        <p
          className={`text-${color}-400 text-xs font-semibold uppercase tracking-wide mb-2`}
        >
          {title}
        </p>

        <p className="text-gray-500 text-sm">{emptyText}</p>
      </div>
    );
  }

  const visible = expanded ? factors : factors.slice(0, COLLAPSE_AT);

  const hiddenCount = factors.length - visible.length;

  return (
    <div className="p-5">
      <p
        className={`text-${color}-400 text-xs font-semibold uppercase tracking-wide mb-3`}
      >
        {title}
      </p>

      <ul className="space-y-3">
        {visible.map((f) => (
          <FactorRow
            key={f.group}
            factor={f}
            maxImportance={maxImportance}
            color={color}
          />
        ))}
      </ul>

      {hiddenCount > 0 && (
        <button
          onClick={() => setExpanded(true)}
          className={`text-${color}-400 hover:text-${color}-300 text-xs font-medium mt-3 transition-colors cursor-pointer`}
        >
          Show {hiddenCount} more
        </button>
      )}

      {expanded && factors.length > COLLAPSE_AT && (
        <button
          onClick={() => setExpanded(false)}
          className="text-gray-400 hover:text-gray-300 text-xs font-medium mt-3 ml-3 transition-colors cursor-pointer"
        >
          Show less
        </button>
      )}
    </div>
  );
}

const Main = () => {
  const [redFighter, setRedFighter] = useState("");
  const [blueFighter, setBlueFighter] = useState("");
  const [fighters, setFighters] = useState([]);
  const [redSuggestions, setRedSuggestions] = useState([]);
  const [blueSuggestions, setBlueSuggestions] = useState([]);
  const [result, setResult] = useState(null); // full backend payload
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("Error occurred");

  const animatedConfidence = useCountUp(result?.confidence ?? null);

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
    setResult(null);

    const payload = { red: redFighter.trim(), blue: blueFighter.trim() };
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

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

      if (!data.winner) {
        setIsError(true);
        setErrorMessage("One or both fighters not found. Check the spelling.");
        return;
      }

      setResult(data);
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
  // *Shared maximum importance across BOTH advantages and concerns.*

  const allFactors = [
    ...(result?.factors?.advantages || []),
    ...(result?.factors?.concerns || []),
  ];

  const maxImportance =
    allFactors.length > 0
      ? Math.max(...allFactors.map((f) => f.importance))
      : 0;

  return (
    <div className="min-h-screen flex items-start md:items-center justify-center px-4 py-10 md:py-16 bg-black/60">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8 rounded-xl p-5">
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

        {/* Main prediction card */}
        <div className="bg-[#161616] border border-white/10 rounded-xl overflow-hidden">
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

            <div className="flex md:flex-col items-center justify-center gap-1.5 py-3 md:py-0 md:px-5">
              <div className="border border-white/15 rounded-full w-9 h-9 flex items-center justify-center bg-[#0d0d0d] shrink-0">
                <span className="text-gray-300 text-xs font-bold">VS</span>
              </div>
              {/* <button
                onClick={handleSwap}
                className="text-emerald-400 hover:text-emerald-300 text-xs font-medium transition-colors cursor-pointer"
              >
                swap
              </button> */}
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

          {isError && (
            <div className="border-t border-white/10 bg-red-500/10 px-5 py-3">
              <p className="text-red-300 text-sm text-center font-medium">
                {errorMessage}
              </p>
            </div>
          )}

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

          {/* Prediction + confidence */}
          <div className="border-t border-white/10">
            <div className="grid grid-cols-1 md:grid-cols-2">
              <div className="p-5 text-center md:border-r border-white/10">
                <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide mb-1.5">
                  Predicted winner
                </p>
                <p
                  className={`min-h-[48px] flex items-center justify-center ${
                    result && !isLoading
                      ? "text-emerald-400 font-bold animate-fade-down text-3xl m-4"
                      : "text-gray-500 text-sm m-4"
                  }`}
                >
                  {isLoading
                    ? "Crunching the numbers..."
                    : result
                      ? result.winner
                      : "Enter two fighters above to see a prediction"}
                </p>
              </div>

              <div className="p-5 text-center">
                <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide mb-3">
                  Model confidence
                </p>
                <p className="text-emerald-400 text-4xl font-bold text-center mb-3 tabular-nums min-h-[48px] flex items-center justify-center">
                  {result?.confidence != null && !isLoading
                    ? `${animatedConfidence.toFixed(1)}%`
                    : "—"}
                </p>
                <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-emerald-500 transition-[width] duration-100 ease-out"
                    style={{
                      width:
                        result?.confidence != null && !isLoading
                          ? `${animatedConfidence}%`
                          : "0%",
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Why: dynamic factor breakdown, only shown once we have a result */}
          {result && !isLoading && (
            <div className="border-t border-white/10 grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-white/10">
              <FactorList
                title={`Why ${result.winner} is favored`}
                emptyText="No single factor stands out... this one's close."
                factors={result.factors?.advantages}
                color="emerald"
                maxImportance={maxImportance}
              />
              <FactorList
                title={`Biggest concerns for ${result.winner}`}
                emptyText="No significant red flags identified."
                factors={result.factors?.concerns}
                color="emerald"
                maxImportance={maxImportance}
              />
            </div>
          )}
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
