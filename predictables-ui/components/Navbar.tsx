'use client';

import { useState } from 'react';

import Logo from './Logo';
import Drawer from './Drawer';
import NavbarButtons from './NavbarButtons';
import { data } from '@data/navbarData';

const Navbar = () => {
  const [isDrawerExpanded, setIsDrawerExpanded] = useState(false);

  const handleDrawerButtonClick = () => {
    setIsDrawerExpanded(!isDrawerExpanded);
  };

  const navClasses = `
  w-[100vw] h-fit flex flex-row p-5
  fixed top-0 justify-center items-center`;

  return (
    <>
      <nav className={navClasses}>
        <input id="drawer" type="checkbox" className="drawer-toggle" />
        <div className="drawer-content">
          {/* Page content here */}
          <label htmlFor="drawer" className="drawer-button">
            <div id="navbar-left" className="text-left w-[20%]">
              <Logo onClick={handleDrawerButtonClick} zIndex={1000} />
            </div>
          </label>
          <div id="navbar-right" className="text-right w-[80%]">
            <NavbarButtons data={data} />
          </div>
        </div>
      </nav>
    </>
  );
};

export default Navbar;
