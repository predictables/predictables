import React from "react";
import SingleGraph from "./SingleGraph";

const TwoByTwo = () => {
  return (
    <div className="grid grid-cols-2 grid-rows-2 gap-4">
      <SingleGraph />
      <SingleGraph />
      <SingleGraph />
      <SingleGraph />
    </div>
  );
};

export default TwoByTwo;
