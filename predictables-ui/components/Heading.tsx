import React from "react";

interface HeadingProps {
  text: string;
}

const Heading = ({ text = "Heading needs to be updated" }: HeadingProps) => {
  return <h1 className="text-5xl text-center mt-5 p-10 top-[60px]">{text}</h1>;
};

export default Heading;
