export const dtypeTypes = [
  'string',
  'float',
  'integer',
  'category',
  'date',
  'any',
  'array',
  'boolean',
  'object',
] as const;
type dtypeTypes = (typeof dtypeTypes)[number];

interface DataSeriesTypes {
  values: any[];
  name?: string;
  index?: any[];
  dtype?: dtypeTypes;
}

/**
 * Numerically-typed DataSeries class. See {@link DataSeries} for details.
 *
 * @class DataSeries
 * @property {any[]} values - The values of the DataSeries.
 * @property {string} name - The name of the DataSeries. Defaults to ''.
 * @property {any[]} index - The index of the DataSeries. Defaults to the default index (0, 1, 2, ...).
 * @property {dtypeTypes} dtype - The dtype of the DataSeries. Defaults to 'any'.
 * @property {[number, number]} shape - The shape of the DataSeries.
 *
 * @method {toString} - Returns a string representation of the DataSeries of the following form:
 * `DataSeries([1, 2, 3], name: 'col1', dtype: 'integer', index: [0, 1, 2])`
 * If any of the attributes are null, they will not be included in the string.
 * @method {toLog} - Logs a string representation of the DataSeries to the console.
 * @method {length} - Returns the length of the DataSeries.
 * @method {nRows} - Returns the number of rows in the DataSeries.
 * @method {nCols} - Returns the number of columns in the DataSeries.
 * @method {map} - Returns a new DataSeries with the values mapped by the callback function.
 * @method {filter} - Returns a new DataSeries with the values filtered by the callback function.
 * @method {sort} - Returns a new DataSeries with the values sorted by the callback function.
 * @method {unique} - Returns a new DataSeries with the unique values of the DataSeries.
 * @method {slice} - Returns a new DataSeries with the values sliced by the start and end indices.
 * @method {valueCounts} - Returns a new DataSeries with the value counts of the DataSeries.
 * @method {resetIndex} - Returns a new DataSeries with the index reset to the default index (0, 1, 2, ...).
 *
 * @static {fromObject} - Returns a new DataSeries from an object.
 * @static {fromJSON} - Returns a new DataSeries from a JSON string.
 *
 * @example
 * const ds = new DataSeries({ values: [1, 2, 3] }); // ds gets the default dtype and index
 * ds.toString(); // DataSeries([1, 2, 3], dtype: any, index: [0, 1, 2])
 * ds.shape; // [3, 1]
 *
 * @example
 * const ds = new DataSeries({ values: [1, 2, 3, 4, 5], name: 'col1', dtype: 'integer' }); // ds gets the default index
 * ds.toString(); // DataSeries([1, 2, 3, 4, 5], name: 'col1', dtype: 'integer', index: [0, 1, 2, 3, 4])
 * ds.shape; // [5, 1]
 
 */
class DataSeries {
  values: any[];
  name: string;
  index: any[];
  dtype: dtypeTypes;
  shape: [number, number];

  constructor({ values, name, index, dtype }: DataSeriesTypes) {
    this.values = values;
    this.name = name || '';
    this.index = index || [...Array(values.length).keys()];
    this.dtype = dtype || 'any';
    this.shape = [this.values.length, 1];
  }

  // getters
  get length() {
    return this.values.length;
  }
  get nRows() {
    return this.values.length;
  }
  get nCols() {
    return 1;
  }
  get columns() {
    return [this.name];
  }
  get data() {
    return this.values;
  }
  get dtypes() {
    return [this.dtype];
  }

  /**
   * @method toString
   * @description Returns a string representation of the DataSeries of the following form:
   * `DataSeries([1, 2, 3], name: 'col1', dtype: 'integer', index: [0, 1, 2])`
   * If any of the attributes are null, they will not be included in the string.
   * @returns {string} - A string representation of the DataSeries.
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3] });
   * console.log(ds.toString()); // DataSeries([1, 2, 3])
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3], name: 'col1', dtype: 'integer' });
   * console.log(ds.toString()); // DataSeries([1, 2, 3], name: 'col1', dtype: 'integer')
   * @example
   * const ds = new DataSeries({
   *    values: ['a', 'b', 'c'],
   *    name: 'newCol',
   *    dtype: 'string',
   *    index: [1, 2, 3]
   * });
   * console.log(ds.toString()); // DataSeries(['a', 'b', 'c'], name: 'newCol', dtype: 'string', index: [1, 2, 3])
   *
   */
  toString() {
    const { values, name, dtype, index } = this;
    const nameStr = name === '' ? '' : `, name: '${name}'`;
    const dtypeStr = dtype === 'any' ? '' : `, dtype: '${dtype}'`;
    let indexBool = true;
    index.forEach((i) => {
      if (i !== index[i]) {
        indexBool = false;
      }
    });
    const indexStr = indexBool ? '' : `, index: ${index}`;
    return `DataSeries([${values}]${nameStr}${dtypeStr}${indexStr})`;
  }

  /**
   * @method toLog
   * @description Logs a string representation of the DataSeries to the console.
   * @param {boolean} string - If true, the string representation of the DataSeries will be logged to the console. If false, the DataSeries itself will be logged to the console.
   * @returns {void}
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3] });
   * ds.toLog(); // DataSeries([1, 2, 3])
   */
  toLog(string: boolean = true) {
    if (string) {
      console.log(this.toString());
    } else {
      console.log(this);
    }
  }

  /**
   * @method map
   * @description Returns a new DataSeries with the values mapped by the callback function.
   * @param {function} callback - The callback function to apply to the values.
   * @returns {DataSeries} - A new DataSeries with the values mapped by the callback function.
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3] });
   * const ds2 = ds.map((value) => value * 2);
   * console.log(ds2.values); // [2, 4, 6]
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3] });
   * const ds2 = ds.map((value, index) => value * index);
   * console.log(ds2.values); // [0, 2, 6]
   */
  map(callback: (value: any, index: number, array: any[]) => any) {
    return this.values.map(callback);
  }

  /**
   * @method filter
   * @description Returns a new DataSeries with the values filtered by the callback function.
   * @param {function} callback - The callback function to apply to the values.
   * @returns {DataSeries} - A new DataSeries with the values filtered by the callback function.
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3] });
   * const ds2 = ds.filter((value) => value > 1);
   * console.log(ds2.values); // [2, 3]
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3] });
   * const ds2 = ds.filter((value, index) => index > 1);
   * console.log(ds2.values); // [3]
   */
  filter(callback: (value: any, index: number, array: any[]) => any) {
    return this.values.filter(callback);
  }

  /**
   * @method sort
   * @description Returns a new DataSeries with the values sorted by the callback function.
   * @param {function} callback - The callback function to apply to the values.
   * @returns {DataSeries} - A new DataSeries with the values sorted by the callback function.
   * @example
   * const ds = new DataSeries({ values: [3, 2, 1] });
   * const ds2 = ds.sort((a, b) => a - b);
   * ds2.toString(); // DataSeries([1, 2, 3], index: [2, 1, 0])
   */
  sort(callback: (a: any, b: any) => number) {
    return this.values.sort(callback);
  }

  /**
   * @method unique
   * @description Returns a new DataSeries with the unique values of the DataSeries.
   * @returns {DataSeries} - A new DataSeries with the unique values of the DataSeries.
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3, 1, 2, 3] });
   * const ds2 = ds.unique();
   * ds2.toString(); // DataSeries([1, 2, 3], index: [0, 1, 2])
   * @example
   * const ds = new DataSeries({ values: ['a', 'b', 'c', 'a', 'b', 'c'] });
   * const ds2 = ds.unique();
   * ds2.toString(); // DataSeries(['a', 'b', 'c'], index: [0, 1, 2])
   */
  unique() {
    return [...new Set(this.values)];
  }

  /**
   * @method slice
   * @description Returns a new DataSeries with the values sliced by the start and end indices.
   * @param {number} start - The start index of the slice.
   * @param {number} end - The end index of the slice.
   * @returns {DataSeries} - A new DataSeries with the values sliced by the start and end indices.
   */
  slice(start: number, end: number) {
    return new DataSeries({
      values: this.values.slice(start, end),
      name: this.name,
      index: this.index.slice(start, end),
      dtype: this.dtype,
    });
  }

  /**
   * @method forEach
   * @description Applies a callback function to each value in the DataSeries.
   * @param {function} callback - The callback function to apply to the values.
   * @returns {void}
   */
  forEach(callback: (value: any, index: number, array: any[]) => void) {
    return this.values.forEach(callback);
  }

  /**
   * @method valueCounts
   * @description Returns a new DataSeries with the unique values of the DataSeries as the index and the value counts of the DataSeries as the values.
   * @returns {DataSeries} - A new DataSeries with the unique values of the DataSeries as the index and the value counts of the DataSeries as the values.
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3, 1, 2, 3] });
   * const ds2 = ds.valueCounts();
   * ds.toString(); // DataSeries([1, 2, 3, 1, 2, 3], index: [0, 1, 2, 3, 4, 5])
   * ds2.toString(); // DataSeries([2, 2, 2], index: [1, 2, 3])
   *
   *
   */
  valueCounts() {
    const unique = this.unique();
    const counts = unique.map((value) => {
      return this.values.filter((v) => v === value).length;
    });

    // let outputDtype;
    // if (Math.max(...counts) < 256) {
    //   outputDtype = Uint8Array;
    // } else if (Math.max(...counts) < 65536) {
    //   outputDtype = Uint16Array;
    // } else if (Math.max(...counts) < 4294967296) {
    //   outputDtype = Uint32Array;
    // } else {
    //   outputDtype = Float64Array;
    // }

    const ds = new DataSeries({
      values: counts,
      name: 'valueCounts',
      index: unique,
      dtype: 'integer',
    });
  }

  /**
   * @method resetIndex
   * @description Returns a new DataSeries with the index reset to the default index (0, 1, 2, ...). If no index is set, the index will be set to the default index.
   * @returns {DataSeries} - A new DataSeries with the index set to the default index.
   * @example
   * const ds = new DataSeries({ values: [1, 2, 3], index: [3, 4, 5] });
   * const ds2 = ds.resetIndex();
   * ds.toString(); // DataSeries([1, 2, 3], index: [3, 4, 5])
   * ds2.toString(); // DataSeries([1, 2, 3], index: [0, 1, 2])
   */
  resetIndex() {
    let nm: string = this.name;
    let dt: dtypeTypes = this.dtype;
    let idx: number[] = [...Array(this.values.length).keys()];
    return new DataSeries({
      values: this.values,
      name: nm,
      index: idx,
      dtype: dt,
    });
  }

  /**
   * @static fromObject
   * @description Returns a new DataSeries from an object. Must be able to destructure the object into the following form:
   * `{ values: any[] }`
   * Otherwise, an error will be thrown.
   * Optionally takes a name and dtype, both of which default to '' and 'any', respectively, as well as an index, which defaults to the default index (0, 1, 2, ...).
   * @param {object} obj - The object to create the DataSeries from. Must at least have a "values" attribute, but may also have "name", "index", and "dtype" attributes. If any of the attributes are not included, they will be added before destructuring.
   * @returns {DataSeries} - A new DataSeries created from the object.
   * @note This requires slightly complex type annotations to work properly.
   * @example
   * const ds = DataSeries.fromObject({ values: [1, 2, 3] });
   * ds.toString(); // DataSeries([1, 2, 3], dtype: any, index: [0, 1, 2])
   * @example
   * const ds = DataSeries.fromObject({ values: [1, 2, 3], name: 'col1', dtype: 'integer' });
   * ds.toString(); // DataSeries([1, 2, 3], name: 'col1', dtype: 'integer', index: [0, 1, 2])
   * @example
   * const ds = DataSeries.fromObject({
   *   values: ['a', 'b', 'c'],
   *   name: 'newCol',
   *   dtype: 'string',
   *   index: [1, 2, 3]
   * });
   * ds.toString(); // DataSeries(['a', 'b', 'c'], name: 'newCol', dtype: 'string', index: [1, 2, 3])
   */
  static fromObject(obj: {}) {
    const { values, name, index, dtype } = obj as DataSeriesTypes;
    return new DataSeries({ values, name, index, dtype });
  }

  /**
   * @static fromJSON
   * @description Returns a new DataSeries from a JSON string. Must be able to parse the JSON string into an object of the following form:
   * `{ values: any[] }`
   * Otherwise, an error will be thrown.
   * Optionally takes a name and dtype, both of which default to '' and 'any', respectively, as well as an index, which defaults to the default index (0, 1, 2, ...).
   * @param {string} json - The JSON string to create the DataSeries from. Must at least have a "values" attribute, but may also have "name", "index", and "dtype" attributes. If any of the attributes are not included, they will be added before parsing.
   * @returns {DataSeries} - A new DataSeries created from the JSON string.
   */
  static fromJSON(json: string) {
    const obj = JSON.parse(json);
    return DataSeries.fromObject(obj);
  }
}

export default DataSeries;
