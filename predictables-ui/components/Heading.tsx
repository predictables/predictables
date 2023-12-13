import React from "react";

interface HeadingProps {
  text: string;
  className?: string;
}

const Heading = ({
  text = "Heading needs to be updated",
  className = "p-10",
}: HeadingProps) => {
  return (
    <h1 className={"text-5xl text-center mt-4 top-[60px] " + className}>
      {text}
    </h1>
  );
};

export default Heading;
