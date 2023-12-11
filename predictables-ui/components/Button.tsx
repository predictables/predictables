import React, { ReactNode } from "react";

interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  filledIn?: boolean;
}

const Button = ({ children, onClick, filledIn = true }: ButtonProps) => {
  return (
    <button
      onClick={onClick}
      className={`${filledIn ? `btn btn-primary` : `btn-outline`}
      rounded-3xl delay-100 justify-center items-center p-0 mx-2 min-w-[90px] w-fit
    `}
    >
      {children}
    </button>
  );
};

export default Button;
