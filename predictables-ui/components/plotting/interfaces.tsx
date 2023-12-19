import React from 'react';
import DataSeries from '@models/DataTable/DataSeries';
import DataTable from '@models/DataTable/DataTable';

export interface PlotContainerProps {
  height?: number;
  width?: number;
  children?: React.ReactNode;
}

export interface AxesProps {
  height?: number;
  width?: number;
  offset?: number;
  color?: string;
  includeXaxis?: boolean;
  includeYaxis?: boolean;
}

export interface ScatterProps {
  height?: number;
  width?: number;
  data: DataTable | null | undefined;
  xCol: string;
  yCol: string;
  xScaleType?: string;
  yScaleType?: string;
  colorCol?: string;
  sizeCol?: string;
  marginTop?: number;
  marginRight?: number;
  marginLeft?: number;
  marginBottom?: number;
}
