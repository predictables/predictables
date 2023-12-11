import React from "react";
import Heading from "@components/Heading";
import TwoByTwo from "@components/eda/TwoByTwo";

const EDApage = () => {
  return (
    <section className="items-center h-[100vh] flex flex-col border-2 border-black">
      <Heading text="EDA Page" />
      <TwoByTwo />
    </section>
  );
};

export default EDApage;
