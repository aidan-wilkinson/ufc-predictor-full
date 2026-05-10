import React from "react";
import Main from "./components/Main";
import ParticlesBackground from "./components/ParticlesBackground";

function App() {
  return (
    <>
      <ParticlesBackground />
      <div className="relative z-10">
        <Main />
      </div>
    </>
  );
}

export default App;
