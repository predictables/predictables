// categorical array is a typed array with a map of values to indices
// depending on cardinality, it is a UInt8Array, UInt16Array, or UInt32Array
// with an additional map of values to indices and indices to values

/**
 * @class CategoricalArray
 * @classdesc A class for categorical data arrays. Extends Uint32Array to add categorical-specific functionality. Encodes the data as a map of categories to values. Used for encoding categorical data in a typed array for the CategoricalSeries class.
 * @augments Uint32Array
 * @param {array} values - The values to encode in the array.
 * @method valueMap - Returns a map of the values in the array to their indices.
 * @method indexMap - Returns a map of the indices in the array to their values.
 * @method cardinality - Returns the cardinality of the array.
 * @method fromIndex - Returns the value at the given index.
 * @method fromValue - Returns the index of the given value.
 * @method set - Sets the value at the given index.
 */
class CategoricalArray extends Uint32Array {
  // map of values to indices
  private _valueMap: Map<string, number>;
  // map of indices to values
  private _indexMap: Map<number, string>;
  // cardinality of the array
  private _cardinality: number;

  constructor(values: any[]) {
    super(values.length);
    this._valueMap = new Map();
    this._indexMap = new Map();
    this._cardinality = 0;
    values.forEach((value: any, i: number) => {
      if (!this._valueMap.has(value)) {
        this._valueMap.set(value, this._cardinality);
        this._indexMap.set(this._cardinality, value);
        this._cardinality++;
      }
      this[i] = this._valueMap.get(value)!;
    });
  }

  get valueMap() {
    return this._valueMap;
  }

  get indexMap() {
    return this._indexMap;
  }

  get cardinality() {
    return this._cardinality;
  }

  /**
   * @method isCategoryKnown
   * @description Returns whether or not the given value is known to the array.
   * @param {string} value - The value to check.
   * @returns {boolean} Whether or not the value is known to the array.
   * @memberof CategoricalArray
   * @instance
   * @example
   * const arr = new CategoricalArray(['a', 'b', 'c']);
   * arr.isCategoryKnown('a'); // true
   * arr.isCategoryKnown('d'); // false
   */
  isCategoryKnown(value: string) {
    return this._valueMap.has(value);
  }

  /**
   * @method addCategory
   * @description Adds a category to the array.
   * @param {string} value - The value to add.
   * @param {boolean} [inPlace=false] - Whether or not to return the index of the added value or quietly add the value to the array.
   * @returns {number} The index of the added value.
   * @memberof CategoricalArray
   * @instance
   * @example
   * const arr = new CategoricalArray(['a', 'b', 'c']);
   * arr.addCategory('d'); // 3
   * arr.addCategory('e'); // 4
   * arr.addCategory('a', inplace=true); // undefined
   */
  addCategory(value: string, inPlace: boolean = false) {
    if (!this._valueMap.has(value)) {
      this._valueMap.set(value, this._cardinality);
      this._indexMap.set(this._cardinality, value);
      this._cardinality++;
    } else {
      console.warn(`The value ${value} already exists in the array.`);
    }

    if (!inPlace) {
      return this._valueMap.get(value)!;
    }
  }

  /**
   * @method removeCategory
   * @description Removes a category from the array.
   * @param {string} value - The value to remove.
   * @param {boolean} [inPlace=false] - Whether or not to return the index of the removed value or quietly remove the value from the array.
   * @returns {number} The index of the removed value, or undefined if the value was not found.
   * @memberof CategoricalArray
   * @instance
   * @example
   * const arr = new CategoricalArray(['a', 'b', 'c']);
   * arr.removeCategory('b'); // 1
   * arr.removeCategory('d'); // undefined
   * arr.removeCategory('a', inplace=true); // undefined
   * arr.get(0); // 'c' // 'a' and 'b' have been removed, so the indices have shifted
   */
  removeCategory(value: string, inPlace: boolean = false) {
    if (this._valueMap.has(value)) {
      const index = this._valueMap.get(value)!;
      this._valueMap.delete(value);
      this._indexMap.delete(index);
      this._cardinality--;
      if (!inPlace) {
        return index;
      }
    } else {
      console.warn(`The value ${value} does not exist in the array.`);
    }

    if (!inPlace) {
      return undefined;
    }
  }

  /**
   * @method fromIndex
   * @description Returns the value at the given index. If the index is out of bounds, returns undefined or raises an error depending on the value of the `safe` parameter.
   * @param {number} index - The index of the value to return.
   * @param {boolean} [safe=false] - Whether or not to return undefined if the index is out of bounds.
   * @returns {string} The value at the given index.
   */
  fromIndex(index: number, safe: boolean = false) {
    if (safe) {
      return this._indexMap.get(index);
    } else {
      // check whether the index is in the keys of the index map
      if (this._indexMap.has(index)) {
        return this._indexMap.get(index);
      } else {
        throw new Error(`Index ${index} is out of bounds.`);
      }
    }
  }

  /**
   * @method fromValue
   * @description Returns the index of the given value. If the value is not known to the array, returns undefined or raises an error depending on the value of the `safe` parameter.
   * @param {string} value - The value to return the index of.
   * @param {boolean} [safe=false] - Whether or not to return undefined if the value is not known to the array.
   * @returns {number} The index of the given value.
   */
  fromValue(value: string, safe: boolean = false) {
    if (safe) {
      return this._valueMap.get(value);
    } else {
      // check whether the value is in the keys of the value map
      if (this._valueMap.has(value)) {
        return this._valueMap.get(value);
      } else {
        throw new Error(`Value ${value} is not known to the array.`);
      }
    }
  }
}

export default CategoricalArray;
