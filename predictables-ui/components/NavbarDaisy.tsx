'use client';

import React, { useState } from 'react';

import NavbarButtons from './NavbarButtons';
import { data } from '@data/navbarData';
import DrawerButton from './_drawer/DrawerButton';
import Drawer from './Drawer';

const Navbar = () => {
  const [isDrawerExpanded, setIsDrawerExpanded] = useState(false);

  const handleDrawerButtonClick = () => {
    setIsDrawerExpanded(!isDrawerExpanded);
  };

  const navClasses = `
  h-fit flex flex-row p-5
  fixed top-0 jusify-end align-end items-right right-0 border-black border-2`;
  return (
    <div className={`drawer`}>
      <input id="my-drawer" type="checkbox" className="drawer-toggle" />
      <div className="drawer-content">
        <div id="navbar-right" className={`text-right w-[80%] ${navClasses}`}>
          <NavbarButtons data={data} />
        </div>
        <label htmlFor="my-drawer" className="button-class-from-your-css pt">
          <DrawerButton onClick={handleDrawerButtonClick} />
        </label>
      </div>
      <div className="drawer-side">
        <label htmlFor="my-drawer" className="drawer-overlay"></label>
        {/* Sidebar content goes here */}
        <ul className="menu p-4 overflow-y-auto w-80 bg-base-100 text-base-content">
          <Drawer />
        </ul>
      </div>
    </div>
  );
};

export default Navbar;
