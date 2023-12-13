import type { NextApiRequest, NextApiResponse } from 'next';
import pl from 'nodejs-polars';

export default async function readParquet(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  try {
    // Get the path to the file from the request body
    const { filePath } = req.body;

    // Read the Parquet file
    const df = pl.readParquet(filePath);

    // Convert DataFrame to JSON (or handle it as needed)
    const data = df.toJSON();

    // Send the data back in response
    res.status(200).json(data);
  } catch (error) {
    // Handle any errors
    res.status(500).json({ error: 'Failed to read Parquet file' });
  }

  return res;
}
