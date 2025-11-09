import React, { useState } from "react";

const Main = () => {
  const [redFighter, setRedFighter] = useState("");
  const [blueFighter, setBlueFighter] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);

  const handlePredict = async () => {
    if (!redFighter.trim() || !blueFighter.trim()) {
      setIsError(true);
      return;
    }

    setIsError(false);
    setIsLoading(true);
    setPrediction(null);

    const payload = {
      red: redFighter.trim(),
      blue: blueFighter.trim(),
    };

    try {
      const res = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      // if Flask returned error (HTTP 400)
      if (!res.ok) {
        const err = await res.json();
        setIsError(true);
        setIsLoading(false);
        return;
      }

      const data = await res.json();
      setPrediction(data.message);
      setIsError(false);
    } catch (e) {
      setIsError(true);
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
          Enter full fighter names in their respective corners, click predict,
          and get the expected winner based on fighter stats and performance
          history.
          <br></br>
          <span className="text-[12px] text-gray-400">
            NOTE: This is most accurate for fighters in the same weight class.
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
            <input
              className="bg-white rounded-md p-2 text-center mx-auto mb-4 focus:outline-none focus:ring-2 focus:ring-red-800 duration-100 w-full"
              type="text"
              placeholder="Fighter Name"
              value={redFighter}
              onChange={(e) => setRedFighter(e.target.value)}
            />
          </div>

          <div className="bg-[#1D4ED8] rounded-md p-4 m-4 md:mx-7 lg:mx-10 shadow-xl w-full md:w-[35%] hover:scale-102 duration-200 flex flex-col">
            <p className="text-white text-3xl font-bold text-center mb-2 mx-auto bg-blue-950 rounded-md p-2 px-4 w-fit">
              Blue Corner
            </p>
            <p className="text-white text-sm text-center mb-4 mt-2 mx-auto font-semibold flex-grow">
              The blue corner is typically assigned to the fighter who is ranked
              lower.
            </p>
            <input
              className="bg-white rounded-md p-2 text-center mx-auto mb-4 focus:outline-none focus:ring-2 focus:ring-blue-950 duration-100 w-full"
              type="text"
              placeholder="Fighter Name"
              value={blueFighter}
              onChange={(e) => setBlueFighter(e.target.value)}
            />
          </div>
        </div>

        {isError && (
          <p className="text-red-600 text-center font-bold text-2xl m-4">
            ERROR OCCURED
          </p>
        )}

        <div className="flex flex-col md:flex-row items-center justify-center gap-6 px-4">
          <button
            onClick={handlePredict}
            className="bg-green-600 hover:bg-green-800 text-white font-bold text-xl px-8 py-4 rounded-lg shadow-xl hover:scale-102 cursor-pointer duration-200 w-full md:w-auto m-4"
          >
            {isLoading ? "Predicting..." : "Predict Winner"}
          </button>

          <div className="bg-black/70 rounded-lg p-6 shadow-xl w-full md:w-auto md:min-w-[280px]">
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
