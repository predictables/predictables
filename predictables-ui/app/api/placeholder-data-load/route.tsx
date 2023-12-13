// placeholder API that returns the breast cancer data:
// @data/bc.json

import type { NextApiRequest, NextApiResponse } from 'next';
import breastCancerData from '@data/bc.json';

export const GET = async (req: NextApiRequest, res: NextApiResponse) => {
  return res.status(200).json(breastCancerData);
};

export default GET;
