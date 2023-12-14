'use client';

import React, { useState, useEffect } from 'react';

import NavbarButtons from './NavbarButtons';
import { data } from '@data/navbarData';
import DrawerButton from './_drawer/DrawerButton';
import LoadDataSection from './_drawer/LoadDataSection';
import CloseButton from './CloseButton';

import DataTable from '@models/DataTable/DataTable';
import DataSeries from '@models/DataTable/DataSeries';

import DataTableContext from '@app/_context/DataTableContext';

interface NavbarProps {
  children: React.ReactNode;
}

const Navbar = ({ children }: NavbarProps) => {
  // State of overall drawer
  const [isDrawerExpanded, setIsDrawerExpanded] = useState(false);

  // State of load data accordion inside drawer
  const [isLoadDataExpanded, setIsLoadDataExpanded] = useState(false);
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [loadedData, setLoadedData] = useState(null);
  const [dataKeys, setDataKeys] = useState<string[] | null>(null);
  const [dt, setDT] = useState<DataTable | null>(null);

  const buildData = (keys: string[], data: any) => {
    let outData: any = [];
    keys.forEach((key: string, i: number) => {
      let dataArr: any[] = [];
      data.forEach((datum: any) => {
        // Ensure the key exists in the datum and push its value
        if (datum.hasOwnProperty(key)) {
          dataArr.push(datum[key]);
        }
      });
      let ds = new DataSeries({ values: dataArr, name: key });
      outData.push(ds);
    });

    // console.log('outData:', outData);
    let outData2 = new DataTable(outData);
    // console.log('outData2:', outData2);
    setDT(outData2);

    return outData2;
  };

  useEffect(() => {
    if (loadedData) {
      const k = Object.keys(loadedData['data'][0]);
      setDataKeys(k);
      buildData(k, loadedData['data']);
    }
  }, [loadedData]);

  const handleDrawerButtonClick = () => {
    setIsDrawerExpanded(!isDrawerExpanded);
  };

  const closeDrawer = () => setIsDrawerExpanded(false);

  // Effect to add/remove event listener for the Escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const toggleDrawer = () => setIsDrawerExpanded((i) => !i);

      if (event.key === 'Escape') {
        toggleDrawer();
      } else if (event.key.toLowerCase() === 'd') {
        toggleDrawer();
      }
    };

    // Add event listener
    if (isDrawerExpanded) {
      window.addEventListener('keydown', handleKeyDown);
    }
  }, [isDrawerExpanded]);

  return (
    <div className="drawer">
      <input
        id="my-drawer"
        type="checkbox"
        className="drawer-toggle"
        checked={isDrawerExpanded}
        readOnly
      />

      {isDrawerExpanded && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={closeDrawer}
          style={{ backdropFilter: 'blur(3px)' }}
        ></div>
      )}

      <div id="content-ex-sidebar" className="drawer-content z-30">
        <nav
          id="navbar"
          className="w-[100vw] h-fit fixed top-0 p-5 flex flex-row"
        >
          <label
            id="navbar-left"
            htmlFor="my-drawer"
            className="drawer-button text-left w-[20%] justify-start align-top items-left left-0"
            onClick={handleDrawerButtonClick}
          >
            <DrawerButton />
          </label>
          <div
            id="navbar-right"
            className="text-right w-[80%] jusify-end align-end items-right right-0"
          >
            <NavbarButtons data={data} isDataLoaded={isDataLoaded} />
          </div>
        </nav>
        <DataTableContext.Provider value={{ dt, setDT }}>
          {children}
        </DataTableContext.Provider>
      </div>

      {/* This is the actual sidebar & sidebar content that
      is hidden until you click the .pt button */}
      <div id="sidebar-content" className="drawer-side z-50">
        <ul className="menu p-4 w-80 min-h-full bg-base-200 text-base-content">
          {/* This is the close button */}
          {isDrawerExpanded && (
            <div className="w-full flex justify-end items-end">
              <CloseButton onClick={handleDrawerButtonClick} />
            </div>
          )}
          <LoadDataSection
            status={isDataLoaded}
            setStatus={setIsDataLoaded}
            setLoadedData={setLoadedData}
            isChecked={isLoadDataExpanded}
            setIsChecked={setIsLoadDataExpanded}
          />
        </ul>
      </div>
    </div>
  );
};

export default Navbar;
