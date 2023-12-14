'use client';

import React from 'react';
import { DataTableContextType } from '@app/interfaces';

const DataTableContext = React.createContext<DataTableContextType | null>(null);

export default DataTableContext;
