'use client';

import React, { use } from 'react';
import Heading from '@components/Heading';

interface LogoProps {
  hover?: boolean;
  darkMode?: boolean;
  inclIcon?: boolean;
  inclTitle?: boolean;
  clickable?: boolean;
  onClick?: () => void;
  zIndex?: number;
}

enum LogoColor {
  BLACK = 'black',
  WHITE = 'white',
}

const LogoHeading = () => {
  return (
    <div className="flex mx-0 px-0 items-center select-none">
      <Heading text="Predic" className="" />
      <span>
        <Heading text="Tables" className="font-medium" />
      </span>
    </div>
  );
};

const LogoIcon = ({
  hover = true,
  darkMode = false,
  clickable = false,
  onClick = () => {},
}: LogoProps) => {
  const logoColor = darkMode ? LogoColor.WHITE : LogoColor.BLACK;
  const hoverColor = darkMode ? LogoColor.BLACK : LogoColor.WHITE;
  const bgCol = darkMode ? 'bg-black' : 'bg-transparent';
  const LOGO_CLASSES = `
  border-[2px] w-[55px] h-[55px] justify-center items-center shadow-lg
  text-${logoColor} border-${logoColor} ${bgCol}
  select-none
  ${hover ? `hover:bg-${logoColor} hover:text-${hoverColor} duration-200` : ''}
  ${clickable ? 'cursor-default active:scale-90' : ''}
  `;
  return (
    <div className={LOGO_CLASSES}>
      <h2
        className={`font-light text-xl text-center justify-center h-[100%] w-[100%] text-${logoColor} 
        ${hover ? `hover:text-${hoverColor}` : ''}
      `}
        onClick={onClick}
      >
        .pt
      </h2>
    </div>
  );
};

const Logo = ({
  hover = true,
  darkMode = false,
  inclIcon = true,
  inclTitle = false,
  onClick = () => {},
  zIndex = 0,
}: LogoProps) => {
  return (
    <div
      className={`flex justify-left items-center
                  ${zIndex ? `z-${zIndex}` : ''}
                `}
    >
      {inclIcon && (
        <LogoIcon
          hover={hover}
          darkMode={darkMode}
          clickable={true}
          onClick={onClick}
        />
      )}
      {inclTitle && <LogoHeading />}
    </div>
  );
};

export default Logo;
