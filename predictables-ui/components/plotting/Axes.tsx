import React from 'react';
import { AxesProps } from './interfaces';

const Axes = ({
  height = 500,
  width = 750,
  offset = 20,
  color = 'black',
  includeXaxis = true,
  includeYaxis = true,
}: AxesProps) => {
  const xAxesEndingOffset = width - offset / 2;
  const yAxesEnd = height - offset;
  const yAxesEndingOffset = height - offset / 2;
  return (
    <>
      {includeXaxis ?? (
        <line
          x1={offset / 2}
          y1={yAxesEnd}
          x2={xAxesEndingOffset}
          y2={yAxesEnd}
          stroke={color}
        />
      )}
      {includeYaxis ?? (
        <line
          x1={offset}
          y1={offset}
          x2={offset}
          y2={yAxesEndingOffset}
          stroke={color}
        />
      )}
    </>
  );
};

export default Axes;
