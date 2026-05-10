import React, { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

const style = document.createElement("style");
style.textContent = `@keyframes fadeDown { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }`;
document.head.appendChild(style);

const Main = () => {
  const [redFighter, setRedFighter] = useState("");
  const [blueFighter, setBlueFighter] = useState("");
  const [fighters, setFighters] = useState([]);
  const [redSuggestions, setRedSuggestions] = useState([]);
  const [blueSuggestions, setBlueSuggestions] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("Error Occured");

  // fetch fighters on mount
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

  const handlePredict = async () => {
    if (!redFighter.trim() || !blueFighter.trim()) {
      setIsError(true);
      setErrorMessage("Both fighter names are required.");
      return;
    }

    setIsError(false);
    setIsLoading(true);
    setPrediction(null);

    const payload = {
      red: redFighter.trim(),
      blue: blueFighter.trim(),
    };

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
            "One or both fighters not found. Check for proper spelling.",
          );
        } else if (data.error && data.error.includes("two different")) {
          setErrorMessage("Please enter two different fighters.");
        } else {
          setErrorMessage("Network Error. Please Try Again");
        }
        setIsLoading(false);
        return;
      }

      if (!data.message || data.message.toLowerCase().includes("not found")) {
        setIsError(true);
        setErrorMessage(
          "One or both fighters not found. Check for proper spelling.",
        );
        return;
      }

      setPrediction(data.message);
      setIsError(false);
    } catch (e) {
      clearTimeout(timeout);
      setIsError(true);
      if (e.name === "AbortError") {
        setErrorMessage("Server inactive, try again in 30 seconds.");
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
    <div className="flex justify-center mt-10">
      <div className="bg-black/80 p-6 rounded-lg shadow-lg w-[90%] md:w-[85%] lg:w-[75%] xl:w-[60%] mb-20 hover:scale-101 duration-300">
        <h1 className="text-white text-4xl font-bold text-center bg-black/90 p-4 rounded-xl mb-0.5 w-[90%] md:w-[85%] lg:w-[82%] mx-auto">
          UFC FIGHT PREDICTOR
        </h1>
        <p className="text-white text-l text-center m-2 w-[90%] md:w-[85%] lg:w-[82%] mx-auto">
          Enter fighter names in their respective corners, click predict, and
          get the expected winner based on fighter stats and performance
          history. <br></br> Historically, the model is 73-74% accurate.
          <br></br>
          <span className="text-[12px] text-gray-400">
            NOTE: If you mix weight classes, results are pound for pound.
          </span>
        </p>
        <div className="flex flex-row flex-wrap justify-center items-stretch">
          <div className="bg-[#F54927] rounded-md p-4 m-4 md:mx-7 lg:mx-10 shadow-xl w-full md:w-[35%] hover:scale-102 duration-200 flex flex-col">
            <p className="text-white text-3xl font-bold text-center mb-2 mx-auto bg-red-800 rounded-md p-2 px-4 w-fit">
              Red Corner
            </p>
            <p className="text-white text-sm text-center mb-4 mt-2 mx-auto font-semibold flex-grow">
              The red corner is typically assigned to the fighter who is ranked
              higher.
            </p>
            <div className="relative w-full">
              <input
                className="bg-white rounded-md p-2 text-center mx-auto mb-4 focus:outline-none focus:ring-2 focus:ring-red-800 duration-100 w-full"
                type="text"
                placeholder="Fighter Name"
                value={redFighter}
                onChange={handleRedChange}
              />
              {redSuggestions.length > 0 && (
                <ul
                  style={{ animation: "0.15s ease-in-out fadeDown" }}
                  className="absolute z-10 bg-white w-full rounded-md shadow-lg max-h-48 overflow-y-auto"
                >
                  {redSuggestions.map((name) => (
                    <li
                      key={name}
                      className="p-2 hover:bg-gray-100 cursor-pointer text-black text-center capitalize border-b border-gray-100 last:border-0"
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

          <div className="bg-[#1D4ED8] rounded-md p-4 m-4 md:mx-7 lg:mx-10 shadow-xl w-full md:w-[35%] hover:scale-102 duration-200 flex flex-col">
            <p className="text-white text-3xl font-bold text-center mb-2 mx-auto bg-blue-950 rounded-md p-2 px-4 w-fit">
              Blue Corner
            </p>
            <p className="text-white text-sm text-center mb-4 mt-2 mx-auto font-semibold flex-grow">
              The blue corner is typically assigned to the fighter who is ranked
              lower.
            </p>
            <div className="relative w-full">
              <input
                className="bg-white rounded-md p-2 text-center mx-auto mb-4 focus:outline-none focus:ring-2 focus:ring-blue-950 duration-100 w-full"
                type="text"
                placeholder="Fighter Name"
                value={blueFighter}
                onChange={handleBlueChange}
              />
              {blueSuggestions.length > 0 && (
                <ul
                  style={{ animation: "0.15s ease-in-out fadeDown" }}
                  className="absolute z-10 bg-white w-full rounded-md shadow-lg max-h-48 overflow-y-auto"
                >
                  {blueSuggestions.map((name) => (
                    <li
                      key={name}
                      className="p-2 hover:bg-gray-100 cursor-pointer text-black text-center capitalize border-b border-gray-100 last:border-0"
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
          <p className="text-red-600 text-center font-bold text-2xl m-4">
            {errorMessage}
          </p>
        )}

        <div className="flex flex-col md:flex-column items-center justify-center gap-4 px-4">
          <button
            onClick={handlePredict}
            className="bg-green-600 hover:bg-green-800 text-white font-bold text-xl px-8 py-4 rounded-lg shadow-xl hover:scale-102 cursor-pointer duration-200 w-full md:w-auto m-4"
          >
            {isLoading ? "Predicting..." : "Predict Winner"}
          </button>

          <div className="bg-black/70 rounded-lg p-6 shadow-xl w-full md:w-auto md:min-w-[400px]">
            <p className="text-gray-400 text-sm text-center">
              Predicted Winner
            </p>
            <p className="text-white text-2xl font-bold text-center mb-2">
              {isLoading
                ? "Predicting winner..."
                : prediction
                  ? prediction
                  : ""}
            </p>
            {/*
            <div className="bg-green-600 rounded-md py-2 px-4 w-fit mx-auto">
              <p className="text-white font-bold text-lg">Confidence %</p>
            </div>
            */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Main;
