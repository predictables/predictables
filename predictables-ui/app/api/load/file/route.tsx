/**
 * Route for loading data from a file. Used in the sidebar.
 */

import type { NextApiRequest, NextApiResponse } from 'next';
import pl from 'nodejs-polars';

export type TableData = {
  columns: string[];
  data: any[];
};

const loadDataFromFile = (
  req: NextApiRequest,
  res: NextApiResponse,
): NextApiResponse => {
  const { filePath } = req.body;
  const fileExtension = filePath.name.split('.').pop();
  let df: pl.DataFrame;
  switch (fileExtension) {
    case 'csv':
      try {
        df = pl.readCSV(filePath);
      } catch (error) {
        throw new Error(`Failed to read CSV file: ${error}`);
      }
      break;
    case 'parquet':
      try {
        df = pl.readParquet(filePath);
      } catch (error) {
        throw new Error(`Failed to read Parquet file: ${error}`);
      }
      break;
    default:
      throw new Error(`Unsupported file type: ${fileExtension}`);
  }

  const data: TableData = {
    columns: df.columns,
    data: df.toJSON(),
  };

  return res.status(200).json(data);
};

export default loadDataFromFile;
