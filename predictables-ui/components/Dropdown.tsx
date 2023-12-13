import React from 'react';
import Link from 'next/link';

interface DropdownProps {
  title: string;
  items: { name: string; route?: string }[];
  isOpen?: boolean;
  handleMouseEnter: () => void;
  handleMouseExit: () => void;
  isClicked?: boolean;
  leftAlign?: boolean;
}

// Dropdown component
const Dropdown = ({
  title,
  items,
  isOpen = false,
  handleMouseEnter,
  handleMouseExit,
  isClicked = false,
  leftAlign = true,
}: DropdownProps) => {
  return (
    <details
      className={`
        dropdown
        ${isClicked ? 'dropdown-open' : ''}
      `}
      id={`${title}-dropdown`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseExit}
    >
      {/* rounded-3xl */}
      <summary
        className={`
        btn rounded-none
        m-1 p-0 mx-2
        justify-center items-center
        min-w-[90px] w-fit
        bg-white text-black border-transparent shadow-md
        duration-100
        hover:bg-black hover:text-white hover:border-transparent `}
      >
        {title}
      </summary>
      <ul
        className={`p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box w-52 
        ${leftAlign ? '' : 'right-0'}
        ${isOpen ? 'block' : 'hidden'}
        `}
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
