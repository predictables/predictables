"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";

interface ItemObj {
  name: string;
  route?: string;
}

interface DropdownProps {
  title: string;
  items: ItemObj[];
  leftAlign?: boolean;
}

// Dropdown component
const Dropdown = ({ title, items, leftAlign = true }: DropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  let inactivityTimer: NodeJS.Timeout | null = null;

  const openDropDown = () => {
    isOpen ? null : setIsOpen(true);
    resetInactivityTimer();
  };

  const closeDropDown = () => {
    isOpen ? setIsOpen(false) : null;
    resetInactivityTimer();
  };

  const resetInactivityTimer = () => {
    clearTimeout(inactivityTimer);
    inactivityTimer = setTimeout(() => {
      closeDropDown();
    }, 5000);
  };

  const handleMouseEnter = () => {
    clearTimeout(inactivityTimer);
  };

  const handleMouseExit = () => {
    resetInactivityTimer();
  };

  useEffect(() => {
    return () => clearTimeout(inactivityTimer);
  }, [inactivityTimer]);

  return (
    <details
      className="dropdown"
      id={`${title}-dropdown`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseExit}
      onClick={resetInactivityTimer}
    >
      <summary className="m-1 btn rounded-3xl delay-100 justify-center items-center p-0 mx-2 min-w-[90px] w-fit">
        {title}
      </summary>
      <ul
        className={`p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box w-52 ${
          leftAlign ? "" : "right-0"
        }`}
      >
        {items.map((item, index) => (
          <li key={index}>
            {item.route && (
              <Link href={item.route}>
                <p className="text-black">{item.name}</p>
              </Link>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
};

export default Dropdown;
