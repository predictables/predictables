// This file contains functions for extracting attributes from DataSeries objects
// DataSeries objects are used to represent columns of data in a DataFrame
// I am implementing an empty DataSeries class in this file for testing & typing purposes

class DataSeries {
  // Blank DataSeries class for testing & typing purposes
  name: string;
  dtype: string;
  index: any[];
  values: any[];

  constructor(name: string, dtype: string, index: any[], values: any[]) {
    this.name = name;
    this.dtype = dtype;
    this.index = index;
    this.values = values;
  }
}

/**
 * Extracts a specified attribute from an array of DataSeries objects.
 *
 * @param {DataSeries[]} values - The array of DataSeries objects to extract the attribute from.
 * @param {string} [attr='name'] - The attribute to extract. This can be 'name', 'dtype', or 'index'.
 *
 * If the attribute is 'name' or 'dtype', the function will loop over the columns of the DataSeries objects and extract the attribute.
 * If the attribute is 'index', the function will check whether there is a common index across all the DataSeries objects.
 * If there is a common index, it will return the first index. If not, it will return an empty array.
 *
 * @returns {string[]} An array of the extracted attributes, or an empty array if the attribute is 'index' and there is no common index.
 *
 * @example
 * const dataSeriesArray = [new DataSeries('name1', 'dtype1', 'index1'), new DataSeries('name2', 'dtype2', 'index2')];
 * extract(dataSeriesArray, 'name'); // Returns ['name1', 'name2']
 */
export const extract = (values: DataSeries[], attr: string = 'name') => {
  // Extracts an attribute from an array of DataSeries objects
  let attrs: string[] = [];

  if (attr == 'name' || attr == 'dtype') {
    // name and dtype are column-level attributes, so we need to loop over the columns
    values.map((value: DataSeries) => {
      attrs.push(value[attr]);
    });
  } else if (attr == 'index') {
    // index is a row-level attribute, so we need to loop over the columns to see whether there is a common index
    let indices: any[] = [];
    values.map((value: DataSeries) => {
      indices.push(value[attr]);
    });

    // check whether all indices are the same
    const allEqual = (arr: any[]) => arr.every((v) => v === arr[0]);

    // if all indices are the same, return the first one
    // otherwise, return an empty array
    if (allEqual(indices)) {
      attrs = indices[0];
    } else {
      attrs = [];
    }
    return attrs;
  }
};
