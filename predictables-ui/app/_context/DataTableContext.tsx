'use client';

import React, { useState, ReactNode, createContext } from 'react';
import { DataTableContextType, DataTableProviderType } from '@app/interfaces';

const DataTableContext = createContext<DataTableContextType | null>(null);

export const DataTableProvider = ({ children }: DataTableProviderType) => {
  const [dt, setDT] = useState<any>(null);

  const updateData = (data: any) => {
    setDT(data);
  };

  return (
    <DataTableContext.Provider value={{ dt, updateData }}>
      {children}
    </DataTableContext.Provider>
  );
};

export default DataTableContext;
